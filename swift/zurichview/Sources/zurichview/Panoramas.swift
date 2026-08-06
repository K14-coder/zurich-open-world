import Foundation
import Metal
import CoreGraphics
import ImageIO
import simd

/// Posed 360° panoramas, projected onto the world at render time.
///
/// This is the opposite approach to façade reconstruction, and it is the one
/// that reaches photographic quality. Rather than rebuilding a flat texture from
/// many photographs — which needs a plane fit, cross-view registration and
/// compositing, and loses sharpness at every step — a single photograph is
/// projected onto whatever geometry is there, from the pose it was taken at.
///
/// There is only ever one image in play, so none of those stages exist, and the
/// pixels arrive at full resolution. Geometry error stops being fatal too: a
/// wall a metre out of place merely slides the texture slightly, instead of
/// breaking a fit.
///
/// Imagery © Mapillary contributors, CC-BY-SA.
struct Panoramas {
    struct Pose {
        var position: SIMD3<Float>
        /// World→camera rotation, as three rows.
        var rotation: simd_float3x3
    }

    let texture: MTLTexture          // 2D array, one layer per panorama
    private(set) var poses: [Pose] = []

    struct JSONFile: Decodable {
        struct Pano: Decodable {
            let file: String
            let pos: [Double]
            let R: [Double]
            let index: Int
        }
        let width: Int
        let height: Int
        let panoramas: [Pano]
    }

    init?(indexURL: URL, device: MTLDevice, queue: MTLCommandQueue) {
        guard let data = try? Data(contentsOf: indexURL),
              let file = try? JSONDecoder().decode(JSONFile.self, from: data),
              !file.panoramas.isEmpty else { return nil }

        let dir = indexURL.deletingLastPathComponent().appendingPathComponent("panoramas")
        let sorted = file.panoramas.sorted { $0.index < $1.index }

        let desc = MTLTextureDescriptor()
        desc.textureType = .type2DArray
        desc.pixelFormat = .rgba8Unorm_srgb
        desc.width = file.width
        desc.height = file.height
        desc.arrayLength = sorted.count
        // Mipmaps matter here: a panorama covers the whole sphere, so most of it
        // is heavily minified on screen at any moment.
        desc.mipmapLevelCount = Int(log2(Double(min(file.width, file.height)))) + 1
        desc.usage = [.shaderRead]
        desc.storageMode = .managed
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }

        var loaded = 0
        for (slice, pano) in sorted.enumerated() {
            let url = dir.appendingPathComponent(pano.file)
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
            let w = file.width, h = file.height
            var pixels = [UInt8](repeating: 0, count: w * h * 4)
            guard let ctx = CGContext(data: &pixels, width: w, height: h,
                                      bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { continue }
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))

            // Pack the segmentation into alpha, which the RGBA texture already
            // allocates and nothing else uses. The renderer needs to know what
            // kind of thing each direction holds, and a parallel texture would
            // double the memory for one byte per pixel.
            let maskURL = dir.appendingPathComponent(
                (pano.file as NSString).deletingPathExtension + "_mask.png")
            if let msrc = CGImageSourceCreateWithURL(maskURL as CFURL, nil),
               let mimg = CGImageSourceCreateImageAtIndex(msrc, 0, nil) {
                var mask = [UInt8](repeating: 255, count: w * h)
                if let mctx = CGContext(data: &mask, width: w, height: h,
                                        bitsPerComponent: 8, bytesPerRow: w,
                                        space: CGColorSpaceCreateDeviceGray(),
                                        bitmapInfo: CGImageAlphaInfo.none.rawValue) {
                    mctx.draw(mimg, in: CGRect(x: 0, y: 0, width: w, height: h))
                    for i in 0..<(w * h) { pixels[i * 4 + 3] = mask[i] }
                }
            }

            tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                        slice: slice, withBytes: pixels,
                        bytesPerRow: w * 4, bytesPerImage: w * h * 4)

            guard pano.pos.count == 3, pano.R.count == 9 else { continue }
            let r = pano.R.map { Float($0) }
            // simd_float3x3 takes columns; the exported matrix is row-major.
            let rot = simd_float3x3(columns: (
                SIMD3(r[0], r[3], r[6]),
                SIMD3(r[1], r[4], r[7]),
                SIMD3(r[2], r[5], r[8])))
            poses.append(Pose(position: SIMD3(Float(pano.pos[0]), Float(pano.pos[1]),
                                              Float(pano.pos[2])),
                              rotation: rot))
            loaded += 1
        }
        guard loaded > 0 else { return nil }

        if let cb = queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: tex)
            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }
        texture = tex
    }

    /// The two panoramas nearest a point, with a blend weight between them.
    ///
    /// Two rather than one so driving past a capture point crossfades instead of
    /// snapping — the same thing Street View does between bubbles, except
    /// continuous.
    func nearest(to p: SIMD3<Float>) -> (a: Int, b: Int, blend: Float) {
        guard poses.count > 1 else { return (0, 0, 0) }
        var order = poses.indices.map { ($0, simd_distance(poses[$0].position, p)) }
        order.sort { $0.1 < $1.1 }
        let (i0, d0) = order[0]
        let (i1, d1) = order[1]
        let total = d0 + d1
        let blend = total > 1e-4 ? d0 / total : 0     // 0 = all of a, 1 = all of b
        return (i0, i1, min(max(blend, 0), 1))
    }
}
