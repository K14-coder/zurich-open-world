import Foundation
import Metal
import simd
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

struct Uniforms {
    var viewProj: simd_float4x4
    var lightViewProj: simd_float4x4
    var sunDirection: SIMD4<Float>
    var cameraPosition: SIMD4<Float>
    var fog: SIMD4<Float>           // rgb + density
    var skyTop: SIMD4<Float>
    var skyHorizon: SIMD4<Float>
    var orthoExtent: SIMD4<Float>   // minX, minZ, maxX, maxZ
    var flags: SIMD4<Float>         // x = ortho bound, y = viewport height
}

struct Camera {
    var eye: SIMD3<Float>
    var target: SIMD3<Float>
    var fovDegrees: Float = 60
    var near: Float = 0.35
    var far: Float = 6000
}

/// Draws the world, offscreen or into a live view through the same path, so what
/// you drive through and what gets written to a PNG are the same picture.
final class Renderer {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let scenePipeline: MTLRenderPipelineState
    private let skyPipeline: MTLRenderPipelineState
    private let shadowPipeline: MTLRenderPipelineState
    private let sceneDepth: MTLDepthStencilState
    private let skyDepth: MTLDepthStencilState

    private let vertexBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer
    private let indexCount: Int
    private let shadowCasterRange: Range<Int>
    private var ortho: Ortho?
    private var materials: Materials?
    private var facadeAtlas: MTLTexture?
    private var sampler: MTLSamplerState!
    private var shadowSampler: MTLSamplerState!
    private let shadowMap: MTLTexture

    var metalDevice: MTLDevice { device }
    var commandQueue: MTLCommandQueue { queue }

    static let sampleCount = 4
    /// Half-extent of the shadow cascade, metres. One cascade is enough because
    /// a driver never sees far before a building occludes the view.
    private let shadowRadius: Float = 340
    private let shadowResolution = 4096

    // A hazy summer afternoon rather than a hard blue sky — haze gives a flat
    // city depth, and hides the edge of the world for free.
    private let skyTop = SIMD3<Float>(0.28, 0.46, 0.72)
    private let skyHorizon = SIMD3<Float>(0.74, 0.81, 0.88)
    private let sunDirection = normalize(SIMD3<Float>(-0.42, 0.62, 0.55))

    init(mesh: WorldMesh) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Fail("no Metal device on this machine")
        }
        guard let queue = device.makeCommandQueue() else { throw Fail("no command queue") }
        self.device = device
        self.queue = queue

        let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        func fn(_ name: String) throws -> MTLFunction {
            guard let f = library.makeFunction(name: name) else {
                throw Fail("shader function \(name) missing")
            }
            return f
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = try fn("scene_vertex")
        desc.fragmentFunction = try fn("scene_fragment")
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.depthAttachmentPixelFormat = .depth32Float
        desc.rasterSampleCount = Self.sampleCount
        scenePipeline = try device.makeRenderPipelineState(descriptor: desc)

        let skyDesc = MTLRenderPipelineDescriptor()
        skyDesc.vertexFunction = try fn("sky_vertex")
        skyDesc.fragmentFunction = try fn("sky_fragment")
        skyDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        skyDesc.depthAttachmentPixelFormat = .depth32Float
        skyDesc.rasterSampleCount = Self.sampleCount
        skyPipeline = try device.makeRenderPipelineState(descriptor: skyDesc)

        let shDesc = MTLRenderPipelineDescriptor()
        shDesc.vertexFunction = try fn("shadow_vertex")
        shDesc.fragmentFunction = nil          // depth only
        shDesc.depthAttachmentPixelFormat = .depth32Float
        shDesc.rasterSampleCount = 1
        shadowPipeline = try device.makeRenderPipelineState(descriptor: shDesc)

        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = true
        sceneDepth = device.makeDepthStencilState(descriptor: dd)!

        let sd = MTLDepthStencilDescriptor()
        sd.depthCompareFunction = .always
        sd.isDepthWriteEnabled = false
        skyDepth = device.makeDepthStencilState(descriptor: sd)!

        let smd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: shadowResolution,
            height: shadowResolution, mipmapped: false)
        smd.usage = [.renderTarget, .shaderRead]
        smd.storageMode = .private
        guard let sm = device.makeTexture(descriptor: smd) else {
            throw Fail("no shadow map")
        }
        shadowMap = sm

        guard let vb = device.makeBuffer(bytes: mesh.vertices,
                                         length: MemoryLayout<Vertex>.stride * mesh.vertices.count,
                                         options: .storageModeShared),
              let ib = device.makeBuffer(bytes: mesh.indices,
                                         length: MemoryLayout<UInt32>.stride * mesh.indices.count,
                                         options: .storageModeShared) else {
            throw Fail("could not upload geometry")
        }
        vertexBuffer = vb
        indexBuffer = ib
        indexCount = mesh.indices.count
        // Only buildings cast. Flat ground writing itself into the shadow map is
        // pure self-shadowing acne — it turned every road pitch black — and a
        // level city gains nothing from terrain casting onto terrain.
        shadowCasterRange = mesh.buildingRange

        let sd2 = MTLSamplerDescriptor()
        sd2.minFilter = .linear
        sd2.magFilter = .linear
        sd2.mipFilter = .linear
        sd2.sAddressMode = .clampToEdge
        sd2.tAddressMode = .clampToEdge
        sd2.maxAnisotropy = 16   // the ground is viewed at a grazing angle constantly
        sampler = device.makeSamplerState(descriptor: sd2)

        let sd3 = MTLSamplerDescriptor()
        sd3.minFilter = .linear
        sd3.magFilter = .linear
        sd3.sAddressMode = .clampToEdge
        sd3.tAddressMode = .clampToEdge
        shadowSampler = device.makeSamplerState(descriptor: sd3)
    }

    func loadOrtho(imageURL: URL, metaURL: URL) {
        ortho = Ortho(imageURL: imageURL, metaURL: metaURL, device: device, queue: queue)
    }

    @discardableResult
    func loadFacadeAtlas(url: URL) -> Bool {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return false }
        let w = img.width, h = img.height
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &pixels, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return false }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm_srgb, width: w, height: h, mipmapped: true)
        desc.usage = [.shaderRead]
        desc.storageMode = .managed
        guard let tex = device.makeTexture(descriptor: desc) else { return false }
        tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                    withBytes: pixels, bytesPerRow: w * 4)
        if let cb = queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: tex)
            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }
        facadeAtlas = tex
        return true
    }

    @discardableResult
    func loadMaterials(directory: URL) -> Bool {
        materials = Materials(directory: directory, device: device, queue: queue)
        return materials != nil
    }

    // MARK: - Encoding

    private func lightMatrix(camera: Camera) -> simd_float4x4 {
        // Centre the cascade a little ahead of the camera: everything behind the
        // driver is off screen, so spending shadow texels on it is waste.
        let ahead = normalize(camera.target - camera.eye)
        let centre = camera.eye + ahead * (shadowRadius * 0.55)
        let lightPos = centre + sunDirection * 900
        let view = lookAt(eye: lightPos, target: centre, up: SIMD3(0, 0, 1))
        let proj = orthographic(left: -shadowRadius, right: shadowRadius,
                                bottom: -shadowRadius, top: shadowRadius,
                                near: 1, far: 2200)
        return proj * view
    }

    /// Shared by the offscreen renderer and the live view.
    func encode(camera: Camera, pass: MTLRenderPassDescriptor,
                width: Int, height: Int, into cb: MTLCommandBuffer) {
        var uniforms = Uniforms(
            viewProj: perspective(fov: camera.fovDegrees * .pi / 180,
                                  aspect: Float(width) / Float(max(1, height)),
                                  near: camera.near, far: camera.far)
                      * lookAt(eye: camera.eye, target: camera.target),
            lightViewProj: lightMatrix(camera: camera),
            sunDirection: SIMD4(sunDirection, 0),
            cameraPosition: SIMD4(camera.eye, 0),
            fog: SIMD4(skyHorizon, 0.00007),
            skyTop: SIMD4(skyTop, 0),
            skyHorizon: SIMD4(skyHorizon, 0),
            orthoExtent: ortho?.extent ?? .zero,
            flags: SIMD4(ortho == nil ? 0 : 1, Float(height),
                         materials == nil ? 0 : 1,
                         Float(materials?.wallLayers ?? 1)))

        // --- Shadow pass ---
        let shadowPass = MTLRenderPassDescriptor()
        shadowPass.depthAttachment.texture = shadowMap
        shadowPass.depthAttachment.loadAction = .clear
        shadowPass.depthAttachment.storeAction = .store
        shadowPass.depthAttachment.clearDepth = 1.0

        if let se = cb.makeRenderCommandEncoder(descriptor: shadowPass) {
            se.setRenderPipelineState(shadowPipeline)
            se.setDepthStencilState(sceneDepth)
            // Front-face culling in the depth pass pushes acne behind surfaces
            // rather than across them; the slope bias mops up the rest.
            se.setCullMode(.front)
            se.setDepthBias(0.0015, slopeScale: 2.0, clamp: 0.01)
            se.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            se.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            se.drawIndexedPrimitives(
                type: .triangle, indexCount: shadowCasterRange.count,
                indexType: .uint32, indexBuffer: indexBuffer,
                indexBufferOffset: shadowCasterRange.lowerBound * MemoryLayout<UInt32>.stride)
            se.endEncoding()
        }

        // --- Main pass ---
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.setRenderPipelineState(skyPipeline)
        enc.setDepthStencilState(skyDepth)
        enc.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        enc.setRenderPipelineState(scenePipeline)
        enc.setDepthStencilState(sceneDepth)
        enc.setCullMode(.none)
        enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        enc.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentTexture(ortho?.texture, index: 0)
        enc.setFragmentTexture(shadowMap, index: 1)
        enc.setFragmentTexture(materials?.albedo, index: 2)
        enc.setFragmentTexture(materials?.normal, index: 3)
        enc.setFragmentTexture(materials?.roughness, index: 4)
        enc.setFragmentTexture(facadeAtlas, index: 5)
        enc.setFragmentSamplerState(sampler, index: 0)
        enc.setFragmentSamplerState(shadowSampler, index: 1)
        enc.drawIndexedPrimitives(type: .triangle, indexCount: indexCount,
                                  indexType: .uint32, indexBuffer: indexBuffer,
                                  indexBufferOffset: 0)
        enc.endEncoding()
    }

    // MARK: - Offscreen

    func render(camera: Camera, width: Int, height: Int) throws -> CGImage {
        let msaaDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        msaaDesc.textureType = .type2DMultisample
        msaaDesc.sampleCount = Self.sampleCount
        msaaDesc.usage = .renderTarget
        msaaDesc.storageMode = .private

        let resolveDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        resolveDesc.usage = [.renderTarget, .shaderRead]
        resolveDesc.storageMode = .managed

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false)
        depthDesc.textureType = .type2DMultisample
        depthDesc.sampleCount = Self.sampleCount
        depthDesc.usage = .renderTarget
        depthDesc.storageMode = .private

        guard let msaa = device.makeTexture(descriptor: msaaDesc),
              let resolve = device.makeTexture(descriptor: resolveDesc),
              let depth = device.makeTexture(descriptor: depthDesc) else {
            throw Fail("could not make render targets")
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = msaa
        pass.colorAttachments[0].resolveTexture = resolve
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .multisampleResolve
        pass.colorAttachments[0].clearColor =
            MTLClearColor(red: 0.74, green: 0.81, blue: 0.88, alpha: 1)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1.0

        guard let cb = queue.makeCommandBuffer() else { throw Fail("no command buffer") }
        encode(camera: camera, pass: pass, width: width, height: height, into: cb)
        if let blit = cb.makeBlitCommandEncoder() {
            blit.synchronize(resource: resolve)
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        return try image(from: resolve, width: width, height: height)
    }

    private func image(from texture: MTLTexture, width: Int, height: Int) throws -> CGImage {
        let rowBytes = width * 4
        var pixels = [UInt8](repeating: 0, count: rowBytes * height)
        pixels.withUnsafeMutableBytes { raw in
            texture.getBytes(raw.baseAddress!, bytesPerRow: rowBytes,
                             from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let img = CGImage(width: width, height: height, bitsPerComponent: 8,
                                bitsPerPixel: 32, bytesPerRow: rowBytes,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGBitmapInfo(rawValue:
                                    CGImageAlphaInfo.noneSkipFirst.rawValue
                                    | CGBitmapInfo.byteOrder32Little.rawValue),
                                provider: provider, decode: nil,
                                shouldInterpolate: false, intent: .defaultIntent) else {
            throw Fail("could not build image")
        }
        return img
    }

    static func write(_ image: CGImage, to url: URL) throws {
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw Fail("could not open \(url.lastPathComponent)")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw Fail("png write failed") }
    }

    // MARK: - Shaders

    static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Uniforms {
        float4x4 viewProj;
        float4x4 lightViewProj;
        float4 sunDirection;
        float4 cameraPosition;
        float4 fog;
        float4 skyTop;
        float4 skyHorizon;
        float4 orthoExtent;
        float4 flags;       // x = ortho bound, y = viewport height,
                            // z = materials bound, w = wall layer count
    };

    // float3, NOT packed_float3: SIMD3<Float> is 16-byte aligned in Swift, so
    // this struct is 64 bytes. packed_float3 would make it 52 and every vertex
    // would be read from the wrong offset.
    struct Vertex {
        float3 position;
        float3 normal;
        float4 colour;   // rgb + material id
        float4 params;   // building base, storey height, seed, building height
    };

    struct VOut {
        float4 clip [[position]];
        float3 world;
        float3 normal;
        float3 colour;
        float material;
        float4 params;
    };

    vertex VOut scene_vertex(uint vid [[vertex_id]],
                             device const Vertex* verts [[buffer(0)]],
                             constant Uniforms& u [[buffer(1)]]) {
        Vertex v = verts[vid];
        VOut o;
        o.clip = u.viewProj * float4(v.position, 1.0);
        o.world = v.position;
        o.normal = v.normal;
        o.colour = v.colour.xyz;
        o.material = v.colour.w;
        o.params = v.params;
        return o;
    }

    vertex float4 shadow_vertex(uint vid [[vertex_id]],
                                device const Vertex* verts [[buffer(0)]],
                                constant Uniforms& u [[buffer(1)]]) {
        return u.lightViewProj * float4(verts[vid].position, 1.0);
    }

    static inline float hash11(float n) {
        return fract(sin(n * 12.9898) * 43758.5453);
    }

    /// Fraction of the sun reaching this point. 3x3 PCF over the cascade.
    static float sunVisibility(float3 world, float ndl,
                               constant Uniforms& u,
                               depth2d<float> shadowMap, sampler smp) {
        float4 lp = u.lightViewProj * float4(world, 1.0);
        float3 proj = lp.xyz / lp.w;
        float2 uv = float2(proj.x * 0.5 + 0.5, -proj.y * 0.5 + 0.5);
        if (uv.x < 0.001 || uv.x > 0.999 || uv.y < 0.001 || uv.y > 0.999) return 1.0;

        // Steeply-lit surfaces need more bias or they shadow themselves.
        float bias = mix(0.0016, 0.0003, ndl);
        float texel = 1.0 / 4096.0;
        float lit = 0.0;
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                float d = shadowMap.sample(smp, uv + float2(i, j) * texel);
                lit += (proj.z - bias <= d) ? 1.0 : 0.0;
            }
        }
        return lit / 9.0;
    }

    /// A Zurich façade, synthesized.
    ///
    /// Street-level photography cannot be used here — it is full of parked cars
    /// and people, which is exactly what this world is meant to be without. So
    /// the façade is built rather than photographed: storey rhythm keyed to the
    /// building's own base, recessed windows with frames and sills, glass that
    /// reflects the sky, a stone plinth, a cornice, and vertical weathering.
    static float3 facade(float3 albedo, float3 world, float3 n, float4 params,
                         float3 viewDir, constant Uniforms& u, thread float& gloss) {
        float base   = params.x;
        float storey = max(2.4, params.y);
        float seed   = params.z;
        float total  = max(3.0, params.w);

        float h = world.y - base;
        // Run the horizontal axis along whichever way the wall faces.
        float along = (abs(n.x) > abs(n.z)) ? world.z : world.x;

        // Ground floor is taller and glassier — shopfronts, not flats.
        float groundH = storey * 1.45;
        bool ground = h < groundH;

        float fy, bay;
        if (ground) {
            fy = clamp(h / groundH, 0.0, 1.0);
            bay = 3.2 + hash11(seed * 3.1) * 0.9;
        } else {
            fy = fract((h - groundH) / storey);
            bay = 2.15 + hash11(seed * 7.3) * 0.75;
        }
        float bayIndex = floor(along / bay);
        float fx = fract(along / bay);

        float wx0 = ground ? 0.10 : 0.20;
        float wx1 = ground ? 0.90 : 0.80;
        float wy0 = ground ? 0.10 : 0.24;
        float wy1 = ground ? 0.88 : 0.84;

        float inX = step(wx0, fx) * step(fx, wx1);
        float inY = step(wy0, fy) * step(fy, wy1);
        float win = inX * inY;

        // No windows in the plinth or immediately under the cornice.
        float capTop = total - 0.9;
        win *= step(0.55, h) * step(h, capTop);

        // Distance inside the opening, used to fake depth: the reveal darkens
        // towards the frame, which is what makes a window read as a hole rather
        // than a painted rectangle.
        float dx = min(fx - wx0, wx1 - fx) / max(0.001, (wx1 - wx0) * 0.5);
        float dy = min(fy - wy0, wy1 - fy) / max(0.001, (wy1 - wy0) * 0.5);
        float inset = clamp(min(dx, dy) * 3.2, 0.0, 1.0);

        // Glass: dark interior plus a sky reflection that strengthens at grazing
        // angles. This is what stops windows looking like grey paint.
        float fres = pow(1.0 - saturate(dot(n, -viewDir)), 3.0);
        float3 sky = mix(u.skyHorizon.xyz, u.skyTop.xyz, 0.35);
        float3 interior = float3(0.055, 0.065, 0.080)
                        + hash11(bayIndex * 3.7 + floor(h / storey) * 11.3) * 0.045;
        float3 glass = mix(interior, sky, 0.18 + 0.62 * fres);

        float bar = step(0.47, fract(fx * 2.0)) * step(fract(fx * 2.0), 0.53);
        glass = mix(glass, albedo * 0.85, bar * 0.55 * win);

        float3 colour = albedo;
        colour = mix(colour, albedo * 0.55, win * (1.0 - inset));
        colour = mix(colour, glass, win * inset);
        gloss = win * inset;

        float sill = inX * step(wy0 - 0.075, fy) * step(fy, wy0 - 0.005);
        colour = mix(colour, albedo * 1.22, sill * 0.9);

        float lintel = inX * step(wy1 + 0.005, fy) * step(fy, wy1 + 0.05);
        colour = mix(colour, albedo * 0.72, lintel * 0.8);

        float plinth = 1.0 - smoothstep(0.55, 0.95, h);
        colour = mix(colour, float3(0.42, 0.41, 0.39), plinth * 0.85);

        float cornice = smoothstep(capTop - 0.35, capTop, h)
                      * (1.0 - smoothstep(total - 0.12, total, h));
        colour = mix(colour, albedo * 1.16, cornice * 0.8);
        float underCornice = smoothstep(capTop - 0.75, capTop - 0.35, h)
                           * (1.0 - smoothstep(capTop - 0.35, capTop - 0.2, h));
        colour *= (1.0 - 0.18 * underCornice);

        float slab = smoothstep(0.0, 0.05, fy) * smoothstep(0.10, 0.05, fy);
        colour *= (1.0 - 0.13 * slab * (1.0 - float(ground)));

        // Vertical weathering, strongest low down.
        float streak = hash11(floor(along * 1.7) + seed * 91.0);
        colour *= 1.0 - 0.07 * streak * (1.0 - smoothstep(0.0, 12.0, h));

        return colour;
    }

    static inline float3 aces(float3 x) {
        const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
        return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
    }

    fragment float4 scene_fragment(VOut in [[stage_in]],
                                   constant Uniforms& u [[buffer(1)]],
                                   texture2d<float> ortho [[texture(0)]],
                                   depth2d<float> shadowMap [[texture(1)]],
                                   texture2d_array<float> matAlbedo [[texture(2)]],
                                   texture2d_array<float> matNormal [[texture(3)]],
                                   texture2d_array<float> matRough [[texture(4)]],
                                   texture2d<float> facadeAtlas [[texture(5)]],
                                   sampler orthoSampler [[sampler(0)]],
                                   sampler shadowSampler [[sampler(1)]]) {
        float3 n = normalize(in.normal);
        float3 sun = normalize(u.sunDirection.xyz);
        float3 viewDir = normalize(in.world - u.cameraPosition.xyz);
        float3 albedo = in.colour;
        float photographic = 0.0;
        float gloss = 0.0;
        float roughness = 0.8;
        bool haveMaterials = u.flags.z > 0.5;

        // Terrain, pavement and roofs are the aerial photograph itself. The
        // ortho is top-down, so every building's real roof already sits in it at
        // that XZ. Note the upper bound: foliage (4) and road (5) live above
        // this range and must not be caught by it.
        bool photoSurface = (in.material > 2.5 && in.material < 3.5)
                         || (in.material > 1.5 && in.material < 2.5);
        if (photoSurface && u.flags.x > 0.5) {
            float2 uv = float2((in.world.x - u.orthoExtent.x)
                                 / (u.orthoExtent.z - u.orthoExtent.x),
                               (in.world.z - u.orthoExtent.y)
                                 / (u.orthoExtent.w - u.orthoExtent.y));
            albedo = ortho.sample(orthoSampler, uv).rgb;
            photographic = 1.0;
        } else if (in.material > 0.5 && in.material < 1.5) {
            // Scanned plaster or concrete under the procedural architecture: the
            // material supplies grain and relief, the façade function supplies
            // the windows, sills and cornices. Neither alone is convincing.
            if (haveMaterials) {
                float along = (abs(n.x) > abs(n.z)) ? in.world.z : in.world.x;
                float2 uv = float2(along, -in.world.y) / 2.4;
                float layer = floor(in.params.z * u.flags.w);
                layer = clamp(layer, 0.0, u.flags.w - 1.0);

                float3 tex = matAlbedo.sample(orthoSampler, uv, uint(layer)).rgb;
                roughness = matRough.sample(orthoSampler, uv, uint(layer)).r;

                // Tangent frame straight from the wall: these are vertical
                // surfaces, so world up crossed with the normal is the tangent.
                float3 T = normalize(cross(float3(0, 1, 0), n));
                float3 B = cross(n, T);
                float3 tn = matNormal.sample(orthoSampler, uv, uint(layer)).rgb * 2.0 - 1.0;
                n = normalize(T * tn.x + B * tn.y + n * max(tn.z, 0.15));

                // Keep the per-building palette tint, but let the photograph
                // carry the detail. Normalised about mid-grey so a dark material
                // does not simply darken the building.
                float lum = max(0.03, dot(tex, float3(0.3333)));
                albedo = albedo * (tex / lum);
            }
            albedo = facade(albedo, in.world, n, in.params, viewDir, u, gloss);
        } else if (in.material > 5.5) {
            // Photographic façade. The plate already contains the building's own
            // shading from the day it was photographed, so it is lit only
            // lightly — relighting it would double every shadow, exactly as with
            // the aerial orthophoto.
            albedo = facadeAtlas.sample(orthoSampler, in.params.xy).rgb;
            photographic = 1.0;
        } else if (in.material > 3.5 && in.material < 4.5) {
            // Foliage.
            //
            // A crown is a 32-triangle blob, and at close range its polygonal
            // outline is unmistakable — worse than no tree at all. Subdividing
            // it into a smooth ball would cost a million triangles across 7,822
            // trees and still read as a green boulder, because the problem is
            // the *silhouette*, not the facets.
            //
            // So punch holes in it instead. Discarding on a leaf-scale noise
            // field makes the outline ragged and lets you see through the crown,
            // which is what a real canopy does. Holes are biased towards the
            // silhouette, where the outline actually shows, leaving the middle
            // of the crown dense.
            float3 cell = floor(in.world * 7.0);
            float leaf = hash11(cell.x * 3.0 + cell.y * 17.0 + cell.z * 29.0);
            float rim = 1.0 - abs(dot(n, viewDir));
            if (leaf < 0.10 + rim * 0.34) discard_fragment();

            albedo *= 0.78 + leaf * 0.44;
            // Light coming through from behind is what makes a canopy glow
            // rather than sit there as a solid object.
            float back = max(0.0, dot(-n, sun));
            albedo += albedo * pow(back, 2.0) * 0.45;
        } else if (in.material > 4.5) {
            // Carriageway: scanned asphalt, which has the aggregate and the
            // subtle relief that no amount of hashing produced.
            if (haveMaterials) {
                uint layer = uint(u.flags.w);      // road sits after the walls
                float2 uv = in.world.xz / 6.0;
                float3 tex = matAlbedo.sample(orthoSampler, uv, layer).rgb;
                roughness = matRough.sample(orthoSampler, uv, layer).r;
                float3 tn = matNormal.sample(orthoSampler, uv, layer).rgb * 2.0 - 1.0;
                // Ground plane: tangent frame is world X and Z.
                n = normalize(float3(tn.x, max(tn.z, 0.30), tn.y));
                float lum = max(0.03, dot(tex, float3(0.3333)));
                albedo = albedo * (tex / lum);
            } else {
                float grain = hash11(floor(in.world.x * 4.7) * 7.0
                                   + floor(in.world.z * 4.7) * 13.0);
                albedo *= 0.975 + grain * 0.05;
            }
            gloss = 0.16;
        }

        float ndl = max(dot(n, sun), 0.0);
        float visibility = sunVisibility(in.world, ndl, u, shadowMap, shadowSampler);

        float3 direct = float3(1.18, 1.09, 0.94) * ndl * visibility;
        float hemi = 0.5 + 0.5 * n.y;
        float3 ambient = mix(float3(0.31, 0.32, 0.34), float3(0.47, 0.52, 0.60), hemi);
        // Shadowed surfaces still see sky, just not sun.
        // Dark materials in canyon shade crush to black otherwise: a shaded
        // street is darker than a sunlit one, not absent.
        ambient *= mix(0.90, 1.0, visibility);

        float3 lit = albedo * (direct + ambient);

        float3 halfway = normalize(-viewDir + sun);
        float specPower = mix(mix(96.0, 12.0, roughness), 180.0, gloss);
        float spec = pow(max(dot(n, halfway), 0.0), specPower)
                   * mix(0.05 + 0.18 * (1.0 - roughness), 0.55, gloss) * visibility;
        lit += spec;

        // An aerial photograph already carries the sun that took it. Relighting
        // doubles the contrast, so fade towards flat albedo — but keep some
        // shadow, or buildings float with nothing beneath them.
        float3 flatLit = albedo * mix(0.72, 1.05, visibility);
        lit = mix(lit, flatLit, photographic * 0.80);

        float dist = length(u.cameraPosition.xyz - in.world);
        float fog = 1.0 - exp(-dist * u.fog.w);
        lit = mix(lit, u.fog.xyz, clamp(fog, 0.0, 1.0));

        return float4(aces(lit * 1.05), 1.0);
    }

    vertex float4 sky_vertex(uint vid [[vertex_id]]) {
        float2 p = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
        return float4(p, 1.0, 1.0);
    }

    fragment float4 sky_fragment(float4 pos [[position]],
                                 constant Uniforms& u [[buffer(1)]]) {
        float t = clamp(pos.y / max(1.0, u.flags.y), 0.0, 1.0);
        float3 c = mix(u.skyTop.xyz, u.skyHorizon.xyz, pow(t, 0.6));
        return float4(aces(c * 1.05), 1.0);
    }
    """
}

struct Fail: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

// MARK: - Matrices

func perspective(fov: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let y = 1 / tan(fov * 0.5)
    let x = y / aspect
    let z = far / (near - far)
    return simd_float4x4(columns: (
        SIMD4(x, 0, 0, 0),
        SIMD4(0, y, 0, 0),
        SIMD4(0, 0, z, -1),
        SIMD4(0, 0, z * near, 0)))
}

func orthographic(left: Float, right: Float, bottom: Float, top: Float,
                  near: Float, far: Float) -> simd_float4x4 {
    simd_float4x4(columns: (
        SIMD4(2 / (right - left), 0, 0, 0),
        SIMD4(0, 2 / (top - bottom), 0, 0),
        SIMD4(0, 0, -1 / (far - near), 0),
        SIMD4(-(right + left) / (right - left),
              -(top + bottom) / (top - bottom),
              -near / (far - near), 1)))
}

func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float> = SIMD3(0, 1, 0))
    -> simd_float4x4 {
    let f = normalize(target - eye)
    var upv = up
    if abs(dot(f, normalize(up))) > 0.999 { upv = SIMD3(1, 0, 0) }
    let s = normalize(cross(f, upv))
    let u = cross(s, f)
    return simd_float4x4(columns: (
        SIMD4(s.x, u.x, -f.x, 0),
        SIMD4(s.y, u.y, -f.y, 0),
        SIMD4(s.z, u.z, -f.z, 0),
        SIMD4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1)))
}
