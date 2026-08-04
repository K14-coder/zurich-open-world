import Foundation
import Metal
import CoreGraphics
import ImageIO

/// Scanned PBR materials for façades and the carriageway.
///
/// Up to this point every surface was invented by arithmetic, which yields a
/// plausible city but never a real-looking one — a hash function has no idea
/// what plaster looks like. These are photographed materials with real albedo,
/// normal and roughness, which is also what finally gives light something to
/// catch on: the geometry is flat, so without normal maps there is nothing for
/// the sun to rake across.
///
/// Materials © ambientCG, CC0.
struct Materials {
    let albedo: MTLTexture
    let normal: MTLTexture
    let roughness: MTLTexture
    let wallLayers: Int
    let roadLayer: Int

    /// Each file is a vertical strip: layer N occupies rows [N*size, (N+1)*size).
    /// Wall layers come first, the road layer last, so one array serves both.
    init?(directory: URL, device: MTLDevice, queue: MTLCommandQueue) {
        guard let wallMeta = Materials.meta(directory / "wall.json"),
              let roadMeta = Materials.meta(directory / "road.json") else { return nil }

        let size = wallMeta.size
        let total = wallMeta.layers + roadMeta.layers

        func build(_ kind: String) -> MTLTexture? {
            guard let wall = Materials.strip(directory / "wall_\(kind).jpg"),
                  let road = Materials.strip(directory / "road_\(kind).jpg") else { return nil }

            let desc = MTLTextureDescriptor()
            desc.textureType = .type2DArray
            desc.pixelFormat = kind == "albedo" ? .rgba8Unorm_srgb : .rgba8Unorm
            desc.width = size
            desc.height = size
            desc.arrayLength = total
            desc.mipmapLevelCount = Int(log2(Double(size))) + 1
            desc.usage = [.shaderRead]
            desc.storageMode = .managed
            guard let tex = device.makeTexture(descriptor: desc) else { return nil }

            var slice = 0
            for (pixels, count) in [(wall, wallMeta.layers), (road, roadMeta.layers)] {
                for layer in 0..<count {
                    let offset = layer * size * size * 4
                    pixels.withUnsafeBytes { raw in
                        tex.replace(region: MTLRegionMake2D(0, 0, size, size),
                                    mipmapLevel: 0, slice: slice,
                                    withBytes: raw.baseAddress!.advanced(by: offset),
                                    bytesPerRow: size * 4,
                                    bytesPerImage: size * size * 4)
                    }
                    slice += 1
                }
            }
            return tex
        }

        guard let a = build("albedo"), let n = build("normal"),
              let r = build("roughness") else { return nil }

        // Without mipmaps a 1024 material minified across a whole façade aliases
        // into crawling noise the moment the camera moves.
        guard let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return nil }
        for tex in [a, n, r] { blit.generateMipmaps(for: tex) }
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        albedo = a; normal = n; roughness = r
        wallLayers = wallMeta.layers
        roadLayer = wallMeta.layers
    }

    private struct Meta: Decodable { let size: Int; let layers: Int }

    private static func meta(_ url: URL) -> Meta? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Meta.self, from: data)
    }

    private static func strip(_ url: URL) -> [UInt8]? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let w = img.width, h = img.height
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &pixels, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return pixels
    }
}

private func / (lhs: URL, rhs: String) -> URL { lhs.appendingPathComponent(rhs) }
