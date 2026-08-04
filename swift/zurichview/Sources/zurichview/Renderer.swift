import Foundation
import Metal
import simd
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

struct Uniforms {
    var viewProj: simd_float4x4
    var sunDirection: SIMD4<Float>
    var cameraPosition: SIMD4<Float>
    var fog: SIMD4<Float>        // rgb + density
    var skyTop: SIMD4<Float>
    var skyHorizon: SIMD4<Float>
    var orthoExtent: SIMD4<Float>   // minX, minZ, maxX, maxZ
    var flags: SIMD4<Float>         // x = 1 when the ortho texture is bound
}

struct Camera {
    var eye: SIMD3<Float>
    var target: SIMD3<Float>
    var fovDegrees: Float = 60
    var near: Float = 0.35
    var far: Float = 6000
}

/// Draws the world offscreen. There is no window here on purpose: a headless
/// render is reproducible, needs no display, and is the only way to check what
/// the city actually looks like from inside an automated build.
final class Renderer {
    var metalDevice: MTLDevice { device }
    var commandQueue: MTLCommandQueue { queue }
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let scenePipeline: MTLRenderPipelineState
    private let skyPipeline: MTLRenderPipelineState
    private let sceneDepth: MTLDepthStencilState
    private let skyDepth: MTLDepthStencilState

    private let vertexBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer
    private let indexCount: Int
    private var ortho: Ortho?
    private var sampler: MTLSamplerState!

    // A hazy Swiss summer afternoon rather than a hard blue sky — haze is what
    // gives a flat city depth, and it hides the edge of the world for free.
    private let skyTop = SIMD3<Float>(0.32, 0.50, 0.74)
    private let skyHorizon = SIMD3<Float>(0.76, 0.82, 0.88)
    private let sunDirection = normalize(SIMD3<Float>(-0.45, 0.72, 0.52))

    init(mesh: WorldMesh) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Fail("no Metal device on this machine")
        }
        guard let queue = device.makeCommandQueue() else { throw Fail("no command queue") }
        self.device = device
        self.queue = queue

        let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        guard let vfn = library.makeFunction(name: "scene_vertex"),
              let ffn = library.makeFunction(name: "scene_fragment"),
              let skyV = library.makeFunction(name: "sky_vertex"),
              let skyF = library.makeFunction(name: "sky_fragment") else {
            throw Fail("shader functions missing")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.depthAttachmentPixelFormat = .depth32Float
        scenePipeline = try device.makeRenderPipelineState(descriptor: desc)

        let skyDesc = MTLRenderPipelineDescriptor()
        skyDesc.vertexFunction = skyV
        skyDesc.fragmentFunction = skyF
        skyDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        skyDesc.depthAttachmentPixelFormat = .depth32Float
        skyPipeline = try device.makeRenderPipelineState(descriptor: skyDesc)

        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = true
        sceneDepth = device.makeDepthStencilState(descriptor: dd)!

        let sd = MTLDepthStencilDescriptor()
        sd.depthCompareFunction = .always
        sd.isDepthWriteEnabled = false
        skyDepth = device.makeDepthStencilState(descriptor: sd)!

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

        let sd2 = MTLSamplerDescriptor()
        sd2.minFilter = .linear
        sd2.magFilter = .linear
        sd2.mipFilter = .linear
        sd2.sAddressMode = .clampToEdge
        sd2.tAddressMode = .clampToEdge
        sd2.maxAnisotropy = 8   // the ground is viewed at a grazing angle constantly
        sampler = device.makeSamplerState(descriptor: sd2)
    }

    func loadOrtho(imageURL: URL, metaURL: URL) {
        ortho = Ortho(imageURL: imageURL, metaURL: metaURL, device: device, queue: queue)
    }

    /// Shared by the offscreen renderer and the live view, so what you drive
    /// through and what gets written to a PNG are the same picture.
    func encode(camera: Camera, pass: MTLRenderPassDescriptor,
                width: Int, height: Int, into cb: MTLCommandBuffer) {
        let aspect = Float(width) / Float(max(1, height))
        var uniforms = Uniforms(
            viewProj: perspective(fov: camera.fovDegrees * .pi / 180, aspect: aspect,
                                  near: camera.near, far: camera.far)
                      * lookAt(eye: camera.eye, target: camera.target),
            sunDirection: SIMD4(sunDirection, 0),
            cameraPosition: SIMD4(camera.eye, 0),
            fog: SIMD4(skyHorizon, 0.00011),
            skyTop: SIMD4(skyTop, Float(height)),
            skyHorizon: SIMD4(skyHorizon, 0),
            orthoExtent: ortho?.extent ?? .zero,
            flags: SIMD4(ortho == nil ? 0 : 1, 0, 0, 0))

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
        enc.setFragmentSamplerState(sampler, index: 0)
        enc.drawIndexedPrimitives(type: .triangle, indexCount: indexCount,
                                  indexType: .uint32, indexBuffer: indexBuffer,
                                  indexBufferOffset: 0)
        enc.endEncoding()
    }

    func render(camera: Camera, width: Int, height: Int) throws -> CGImage {
        let cd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        cd.usage = [.renderTarget, .shaderRead]
        cd.storageMode = .managed
        guard let colour = device.makeTexture(descriptor: cd) else {
            throw Fail("no colour target")
        }

        let dd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false)
        dd.usage = .renderTarget
        dd.storageMode = .private
        guard let depth = device.makeTexture(descriptor: dd) else {
            throw Fail("no depth target")
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = colour
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.76, green: 0.82, blue: 0.88, alpha: 1)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1.0

        guard let cb = queue.makeCommandBuffer() else { throw Fail("no command buffer") }
        encode(camera: camera, pass: pass, width: width, height: height, into: cb)

        if let blit = cb.makeBlitCommandEncoder() {
            blit.synchronize(resource: colour)
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        return try image(from: colour, width: width, height: height)
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
        float4 sunDirection;
        float4 cameraPosition;
        float4 fog;         // rgb, density
        float4 skyTop;
        float4 skyHorizon;
        float4 orthoExtent;
        float4 flags;
    };

    // float3, NOT packed_float3. Swift's SIMD3<Float> is 16-byte aligned, so the
    // Swift Vertex is 48 bytes; packed_float3 here would make this 36 and the
    // shader would read every vertex from the wrong offset.
    struct Vertex {
        float3 position;
        float3 normal;
        float4 colour;   // rgb + material id in w
    };

    struct VOut {
        float4 clip [[position]];
        float3 world;
        float3 normal;
        float3 colour;
        float material;
    };

    vertex VOut scene_vertex(uint vid [[vertex_id]],
                             device const Vertex* verts [[buffer(0)]],
                             constant Uniforms& u [[buffer(1)]]) {
        Vertex v = verts[vid];
        float3 p = float3(v.position);
        VOut o;
        o.clip = u.viewProj * float4(p, 1.0);
        o.world = p;
        o.normal = float3(v.normal);
        o.colour = v.colour.xyz;
        o.material = v.colour.w;
        return o;
    }

    fragment float4 scene_fragment(VOut in [[stage_in]],
                                   constant Uniforms& u [[buffer(1)]],
                                   texture2d<float> ortho [[texture(0)]],
                                   sampler orthoSampler [[sampler(0)]]) {
        float3 n = normalize(in.normal);
        float3 sun = normalize(u.sunDirection.xyz);
        float3 albedo = in.colour;
        float photographic = 0.0;

        // Terrain is the aerial photograph itself — and so are the roofs. The
        // ortho is a top-down image, so the real roof of every building is
        // already sitting in it at that XZ. Sampling it there costs nothing and
        // replaces 12,538 invented flat-brown lids with the actual roofs, tiles,
        // skylights, courtyards and all.
        //
        // It is an approximation: the photo is orthorectified to the ground, so
        // a tall building leans away from nadir and its roof pixels sit slightly
        // off. At Zurich's 15 m average that is a metre or two, well below what
        // you can see from a car.
        bool photoSurface = (in.material > 2.5)                       // terrain
                         || (in.material > 1.5 && in.material < 2.5); // roofs
        if (photoSurface && u.flags.x > 0.5) {
            float2 uv = float2((in.world.x - u.orthoExtent.x)
                                 / (u.orthoExtent.z - u.orthoExtent.x),
                               (in.world.z - u.orthoExtent.y)
                                 / (u.orthoExtent.w - u.orthoExtent.y));
            albedo = ortho.sample(orthoSampler, uv).rgb;
            photographic = 1.0;
        }

        // Procedural facades. Blank extrusions read as cardboard no matter how
        // good the lighting is; a storey rhythm is what makes a box become a
        // building. Storey height matches the 3.2 m used to estimate heights.
        if (in.material > 0.5 && in.material < 1.5) {
            // Run the horizontal axis along whichever way the wall faces.
            float along = (abs(n.x) > abs(n.z)) ? in.world.z : in.world.x;
            float storey = 3.2;
            float bay = 2.6;

            float fy = fract(in.world.y / storey);
            float fx = fract(along / bay);

            // Window pane occupies the middle of each bay and the upper part of
            // each storey, leaving a sill and a spandrel band.
            float win = step(0.16, fx) * step(fx, 0.78)
                      * step(0.30, fy) * step(fy, 0.86);

            // Ground floor is shopfront in central Zurich: taller and glassier.
            float groundFloor = 1.0 - step(storey, in.world.y - 0.0);

            float3 glass = float3(0.16, 0.19, 0.24);
            // A little variety so every pane is not identically lit.
            float jitter = fract(sin(floor(along / bay) * 12.9898
                                   + floor(in.world.y / storey) * 78.233) * 43758.5453);
            glass += jitter * 0.10;

            albedo = mix(albedo, glass, win * (0.72 + 0.2 * groundFloor));

            // Thin darker line at each floor slab.
            float slab = smoothstep(0.0, 0.06, fy) * smoothstep(0.12, 0.06, fy);
            albedo *= (1.0 - 0.22 * slab);
        }

        // Direct sun, plus a sky/ground hemisphere term. The hemisphere light is
        // what keeps north-facing walls and street canyons from going pure black
        // without needing any global illumination.
        float ndl = max(dot(n, sun), 0.0);
        float3 direct = float3(1.05, 1.00, 0.92) * ndl;
        float hemi = 0.5 + 0.5 * n.y;
        float3 ambient = mix(float3(0.34, 0.35, 0.34), float3(0.50, 0.56, 0.64), hemi);

        float3 lit = albedo * (direct + ambient);

        // An aerial photograph already carries the sun that took it. Lighting it
        // again doubles the contrast and turns every roof into a highlight, so
        // fade towards flat albedo for photographic surfaces.
        lit = mix(lit, albedo * 1.06, photographic * 0.82);

        // Cheap specular sheen on near-horizontal surfaces reads as damp tarmac.
        float3 view = normalize(u.cameraPosition.xyz - in.world);
        float3 halfway = normalize(view + sun);
        float spec = pow(max(dot(n, halfway), 0.0), 48.0) * 0.06 * step(0.9, n.y);
        lit += spec;

        float dist = length(u.cameraPosition.xyz - in.world);
        float fog = 1.0 - exp(-dist * u.fog.w);
        lit = mix(lit, u.fog.xyz, clamp(fog, 0.0, 1.0));

        return float4(lit, 1.0);
    }

    vertex float4 sky_vertex(uint vid [[vertex_id]]) {
        // One oversized triangle covering the viewport.
        float2 p = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
        return float4(p, 1.0, 1.0);
    }

    fragment float4 sky_fragment(float4 pos [[position]],
                                 constant Uniforms& u [[buffer(1)]]) {
        // Gradient by screen height. Good enough: the horizon band is what sells
        // the haze, and anything above it is barely in frame from a car.
        float t = clamp(pos.y / u.skyTop.w, 0.0, 1.0);
        float3 c = mix(u.skyTop.xyz, u.skyHorizon.xyz, pow(t, 0.65));
        return float4(c, 1.0);
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

func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float> = SIMD3(0, 1, 0))
    -> simd_float4x4 {
    let f = normalize(target - eye)
    let s = normalize(cross(f, up))
    let u = cross(s, f)
    return simd_float4x4(columns: (
        SIMD4(s.x, u.x, -f.x, 0),
        SIMD4(s.y, u.y, -f.y, 0),
        SIMD4(s.z, u.z, -f.z, 0),
        SIMD4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1)))
}
