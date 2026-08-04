import Foundation
import Metal
import CoreGraphics
import ImageIO
import simd

/// The SWISSIMAGE aerial photograph of the city, as a ground texture.
///
/// This is the difference between ground that is *a colour* and ground that is
/// the actual place — real pavement, tram beds, courtyards, parkland, the river
/// and the lake, all photographed rather than invented.
///
/// Tiles were fetched in swisstopo's LV95 grid, which is the same frame the
/// world uses, so the mapping from world XZ to texture UV is a subtraction and
/// a divide. No reprojection, no resampling error.
struct Ortho {
    let texture: MTLTexture
    /// worldMinX, worldMinZ, worldMaxX, worldMaxZ — handed to the shader so it
    /// can derive UVs from world position without any per-vertex data.
    let extent: SIMD4<Float>

    struct Meta: Decodable {
        let level: Int
        let metresPerPixel: Double
        let width: Int, height: Int
        let worldMinX: Double, worldMinZ: Double
        let worldMaxX: Double, worldMaxZ: Double
    }

    init?(imageURL: URL, metaURL: URL, device: MTLDevice, queue: MTLCommandQueue) {
        guard let metaData = try? Data(contentsOf: metaURL),
              let meta = try? JSONDecoder().decode(Meta.self, from: metaData),
              let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            return nil
        }

        let w = image.width, h = image.height
        let rowBytes = w * 4
        var pixels = [UInt8](repeating: 0, count: rowBytes * h)
        guard let ctx = CGContext(data: &pixels, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: rowBytes,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: true)
        desc.usage = [.shaderRead]
        desc.storageMode = .managed
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }
        tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                    withBytes: pixels, bytesPerRow: rowBytes)

        // Mipmaps are not optional here: a 7680 px texture minified into a few
        // hundred pixels of road surface aliases into crawling noise without them.
        guard let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return nil }
        blit.generateMipmaps(for: tex)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        texture = tex
        extent = SIMD4(Float(meta.worldMinX), Float(meta.worldMinZ),
                       Float(meta.worldMaxX), Float(meta.worldMaxZ))
    }
}
