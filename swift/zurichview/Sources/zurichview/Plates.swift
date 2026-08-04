import Foundation
import simd

/// Photographic façades, hung in front of the extruded walls.
///
/// Each plate is a quad on the plane the reconstruction fitted — not on the OSM
/// footprint, which is a metre or two out — so the geometry comes from the atlas
/// index rather than from the building mesh. That keeps the two completely
/// decoupled: the renderer needs to know nothing about how a plate was made, and
/// a rebuilt atlas drops straight in.
///
/// Imagery © Mapillary contributors, CC-BY-SA.
extension WorldMesh {

    struct AtlasJSON: Decodable {
        struct Plate: Decodable {
            let corners: [[Double]]   // bottom-left, bottom-right, top-left, top-right
            let uv: [Double]          // u0, v0, u1, v1
        }
        let size: Int
        let plates: [Plate]
    }

    @discardableResult
    func buildPlates(url: URL) -> Int {
        guard let data = try? Data(contentsOf: url),
              let atlas = try? JSONDecoder().decode(AtlasJSON.self, from: data) else {
            return 0
        }
        let start = indices.count
        var count = 0

        for plate in atlas.plates {
            guard plate.corners.count == 4, plate.uv.count == 4,
                  plate.corners.allSatisfy({ $0.count == 3 }) else { continue }

            let bl = SIMD3<Double>(plate.corners[0][0], plate.corners[0][1], plate.corners[0][2])
            let br = SIMD3<Double>(plate.corners[1][0], plate.corners[1][1], plate.corners[1][2])
            let tl = SIMD3<Double>(plate.corners[2][0], plate.corners[2][1], plate.corners[2][2])
            let tr = SIMD3<Double>(plate.corners[3][0], plate.corners[3][1], plate.corners[3][2])

            // Face normal, pointed at the street. The plate was rectified from
            // photographs taken there, so it is only ever seen from that side.
            var edge = br - bl
            edge.y = 0
            let len = simd_length(edge)
            guard len > 1e-6 else { continue }
            edge /= len
            let normal = SIMD3<Float>(Float(-edge.z), 0, Float(edge.x))

            let u0 = Float(plate.uv[0]), v0 = Float(plate.uv[1])
            let u1 = Float(plate.uv[2]), v1 = Float(plate.uv[3])

            // v0 is the top of the plate image, so the top corners take v0.
            let corners: [(SIMD3<Double>, SIMD2<Float>)] = [
                (bl, SIMD2(u0, v1)), (br, SIMD2(u1, v1)),
                (tl, SIMD2(u0, v0)), (tr, SIMD2(u1, v0)),
            ]
            let base = UInt32(vertices.count)
            for (p, uv) in corners {
                vertices.append(Vertex(
                    position: SIMD3(Float(p.x), Float(p.y), Float(p.z)),
                    normal: normal,
                    colour: SIMD3(1, 1, 1),
                    material: .plate,
                    params: SIMD4(uv.x, uv.y, 0, 0)))
            }
            indices += [base, base + 2, base + 1, base + 1, base + 2, base + 3]
            count += 1
        }

        setPlateRange(start..<indices.count)
        return count
    }
}
