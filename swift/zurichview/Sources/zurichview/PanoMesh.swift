import Foundation
import CoreGraphics
import ImageIO
import simd

/// A panorama turned into geometry by its own depth.
///
/// This replaces projecting photographs onto the city model, and it is the fix
/// for both things that approach gets wrong. "Distorted" was a photograph
/// stretched across boxes that are not where its content was. "Separate
/// pictures" was two such projections crossfading, each wrong in a different
/// way, so the seam between them moved as you did.
///
/// Here the geometry *is* the photograph: every pixel sits at the distance it
/// was measured at, so a tree is a tree-shaped piece of the world rather than a
/// tree painted on a wall behind it, and moving the camera produces real
/// parallax instead of a dissolve.
///
/// Imagery © Mapillary contributors, CC-BY-SA.
extension WorldMesh {

    /// Grid resolution across the panorama. 384x192 is about 73k vertices per
    /// panorama; finer looks better standing still and costs memory that 44 of
    /// them cannot spare.
    /// Camera height above the road, matching the depth pass.
    private static var CAMERA_HEIGHT_M: Double { 2.2 }

    private static var gridW: Int { 384 }
    private static var gridH: Int { 192 }

    @discardableResult
    func buildPanoMeshes(indexURL: URL) -> Int {
        guard let data = try? Data(contentsOf: indexURL),
              let file = try? JSONDecoder().decode(Panoramas.JSONFile.self, from: data)
        else { return 0 }
        let dir = indexURL.deletingLastPathComponent().appendingPathComponent("panoramas")
        let ordered = file.panoramas.sorted { $0.index < $1.index }

        var ranges: [Range<Int>] = []
        var built = 0
        for (slice, pano) in ordered.enumerated() {
            let depthURL = dir.appendingPathComponent(
                (pano.file as NSString).deletingPathExtension + "_depth.png")
            guard pano.pos.count == 3, pano.R.count == 9,
                  let depth = Self.loadDepth(depthURL) else {
                ranges.append(0..<0)
                continue
            }
            let start = indices.count
            addPanoMesh(depth: depth, pos: pano.pos, R: pano.R, layer: Float(slice))
            ranges.append(start..<indices.count)
            built += 1
        }
        setPanoMeshRanges(ranges)
        return built
    }

    // MARK: - Mesh

    private func addPanoMesh(depth: (w: Int, h: Int, cm: [UInt16]),
                             pos: [Double], R: [Double], layer: Float) {
        let gw = Self.gridW, gh = Self.gridH
        let origin = SIMD3<Double>(pos[0], pos[1], pos[2])

        // The exported matrix maps world ENU to camera. Going the other way is
        // its transpose, since a rotation's inverse is its transpose.
        let rt = simd_double3x3(rows: [
            SIMD3(R[0], R[3], R[6]),
            SIMD3(R[1], R[4], R[7]),
            SIMD3(R[2], R[5], R[8]),
        ])

        let base = UInt32(vertices.count)
        var metres = [Double](repeating: 0, count: gw * gh)

        for j in 0..<gh {
            let v = (Double(j) + 0.5) / Double(gh)
            let lat = Double.pi / 2 - v * Double.pi
            for i in 0..<gw {
                let u = (Double(i) + 0.5) / Double(gw)
                let lon = u * 2 * Double.pi - Double.pi

                let cam = SIMD3(cos(lat) * sin(lon), -sin(lat), cos(lat) * cos(lon))
                // Camera frame back to ENU, then ENU to our world axes: X east,
                // Y up, Z south.
                let enu = rt * cam
                let dir = SIMD3(enu.x, enu.z, -enu.y)

                let sx = min(depth.w - 1, Int(u * Double(depth.w)))
                let sy = min(depth.h - 1, Int(v * Double(depth.h)))
                var d = Double(depth.cm[sy * depth.w + sx]) / 100.0

                // The nadir is the capture vehicle, and the depth pass rejects
                // it — leaving a hole straight down through which the clear
                // colour shows as a white ring around the driver. Floor it with
                // the road instead: geometry on the ground plane, flagged so it
                // renders as asphalt rather than as a photograph of a bonnet.
                let down = -dir.y
                var isFloor = false
                if down > 0.42 {
                    d = Self.CAMERA_HEIGHT_M / max(down, 1e-3)
                    isFloor = true
                }
                metres[j * gw + i] = d

                let p = origin + dir * (d > 0 ? d : 1000.0)
                vertices.append(Vertex(
                    position: SIMD3(Float(p.x), Float(p.y), Float(p.z)),
                    normal: isFloor ? SIMD3(0, 1, 0)
                                    : SIMD3(Float(-dir.x), Float(-dir.y), Float(-dir.z)),
                    colour: isFloor ? SIMD3(0.20, 0.20, 0.215) : SIMD3(1, 1, 1),
                    material: isFloor ? .road : .panoMesh,
                    params: SIMD4(Float(u), Float(v), layer, Float(d))))
            }
        }

        for j in 0..<(gh - 1) {
            for i in 0..<gw {
                let i1 = (i + 1) % gw            // wrap: a panorama is a cylinder
                let a = j * gw + i, b = j * gw + i1
                let c = (j + 1) * gw + i, e = (j + 1) * gw + i1
                let quad = [metres[a], metres[b], metres[c], metres[e]]

                // Sky, and anything the depth pass rejected, is not a surface.
                if quad.contains(where: { $0 <= 0.01 }) { continue }

                // Do not stitch across a depth discontinuity. A quad spanning a
                // near railing and the building behind it would otherwise become
                // a rubber sheet stretched between them, which is the single
                // ugliest artifact this kind of mesh produces.
                let lo = quad.min()!, hi = quad.max()!
                if hi > lo * 1.22 && hi - lo > 0.9 { continue }

                indices += [base + UInt32(a), base + UInt32(c), base + UInt32(b),
                            base + UInt32(b), base + UInt32(c), base + UInt32(e)]
            }
        }
    }

    // MARK: - Depth

    private static func loadDepth(_ url: URL) -> (w: Int, h: Int, cm: [UInt16])? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let w = img.width, h = img.height
        var buf = [UInt16](repeating: 0, count: w * h)
        let ok = buf.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(
                data: raw.baseAddress, width: w, height: h,
                bitsPerComponent: 16, bytesPerRow: w * 2,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
                    | CGBitmapInfo.byteOrder16Little.rawValue) else { return false }
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? (w, h, buf) : nil
    }
}
