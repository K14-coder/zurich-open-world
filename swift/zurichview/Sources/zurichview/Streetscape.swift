import Foundation
import simd

/// Everything on the street that is not the street.
///
/// This turned out to be the real gap in street-level realism, not façade
/// texture. The city was geometrically correct and completely empty: no trees,
/// no tram rails, no kerbs, no lamps. A real street is dense with objects, and
/// a Zurich one is defined by plane trees and tram wire. All of it is in OSM.
extension WorldMesh {

    struct StreetscapeJSON: Decodable {
        struct Point: Decodable { let p: [Double] }
        struct Tree: Decodable { let p: [Double]; let h: Double; let s: Double }
        struct Rail: Decodable { let p: [[Double]] }
        let trees: [Tree]
        let lamps: [Point]
        let rails: [Rail]
    }

    func buildStreetscape(url: URL) {
        let start = indices.count
        if let data = try? Data(contentsOf: url),
           let scape = try? JSONDecoder().decode(StreetscapeJSON.self, from: data) {
            for tree in scape.trees where tree.p.count == 3 {
                addTree(at: SIMD3(tree.p[0], tree.p[1], tree.p[2]),
                        height: tree.h, seed: Float(tree.s))
            }
            for rail in scape.rails { addRail(rail.p) }
            for lamp in scape.lamps where lamp.p.count == 3 {
                addLamp(at: SIMD3(lamp.p[0], lamp.p[1], lamp.p[2]))
            }
        }
        addPavements()
        setStreetscapeRange(start..<indices.count)
    }

    // MARK: - Trees

    private func addTree(at base: SIMD3<Double>, height: Double, seed: Float) {
        let trunkH = height * 0.38
        let trunkR = max(0.10, height * 0.026)

        // Trunk: a five-sided prism. Round enough at the distance you ever see
        // it from, and a fifth of the cost of a cylinder.
        let bark = SIMD3<Float>(0.24, 0.19, 0.15) * (0.85 + seed * 0.3)
        var ring: [SIMD2<Double>] = []
        for i in 0..<5 {
            let a = Double(i) / 5 * 2 * .pi
            ring.append(SIMD2(cos(a) * trunkR, sin(a) * trunkR))
        }
        for i in 0..<5 {
            let a = ring[i], b = ring[(i + 1) % 5]
            let nrm = normalize(SIMD3<Float>(Float(a.x + b.x), 0, Float(a.y + b.y)))
            let idx = UInt32(vertices.count)
            for (p, y) in [(a, 0.0), (b, 0.0), (a, trunkH), (b, trunkH)] {
                vertices.append(Vertex(
                    position: SIMD3(Float(base.x + p.x), Float(base.y + y),
                                    Float(base.z + p.y)),
                    normal: nrm.x.isNaN ? SIMD3(0, 1, 0) : nrm, colour: bark))
            }
            indices += [idx, idx + 2, idx + 1, idx + 1, idx + 2, idx + 3]
        }

        // Canopy: an octahedron subdivided once, pushed out to a sphere and then
        // knocked about per-vertex so no two crowns are the same shape.
        let crownY = base.y + trunkH
        let radius = height * 0.30
        let green = mix(SIMD3<Float>(0.20, 0.34, 0.15),
                        SIMD3<Float>(0.34, 0.46, 0.21), t: seed)

        var pts: [SIMD3<Double>] = [
            SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0),
            SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1),
        ]
        var faces: [(Int, Int, Int)] = [
            (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
            (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
        ]
        var cache: [Int: Int] = [:]
        func midpoint(_ a: Int, _ b: Int) -> Int {
            let key = a < b ? a * 1000 + b : b * 1000 + a
            if let hit = cache[key] { return hit }
            pts.append(normalize(pts[a] + pts[b]))
            cache[key] = pts.count - 1
            return pts.count - 1
        }
        var subdivided: [(Int, Int, Int)] = []
        for f in faces {
            let a = midpoint(f.0, f.1), b = midpoint(f.1, f.2), c = midpoint(f.2, f.0)
            subdivided += [(f.0, a, c), (a, f.1, b), (c, b, f.2), (a, b, c)]
        }
        faces = subdivided

        let vbase = UInt32(vertices.count)
        for (i, unit) in pts.enumerated() {
            let wobble = 0.74 + Double(hash(Float(i) * 3.1 + seed * 71)) * 0.52
            let p = SIMD3(unit.x * radius * wobble,
                          unit.y * radius * wobble * 1.12,
                          unit.z * radius * wobble)
            // Crowns are lit from above and dark underneath.
            let shade = Float(0.72 + 0.30 * (unit.y * 0.5 + 0.5))
            vertices.append(Vertex(
                position: SIMD3(Float(base.x + p.x), Float(crownY + radius * 0.85 + p.y),
                                Float(base.z + p.z)),
                normal: SIMD3(Float(unit.x), Float(unit.y), Float(unit.z)),
                colour: green * shade, material: .foliage))
        }
        for f in faces {
            indices += [vbase + UInt32(f.0), vbase + UInt32(f.1), vbase + UInt32(f.2)]
        }
    }

    // MARK: - Tram rails

    private func addRail(_ pts: [[Double]]) {
        let gauge = 1.435 / 2      // standard gauge, metres from centreline
        let railW = 0.075
        let steel = SIMD3<Float>(0.20, 0.19, 0.18)
        for k in 0..<(pts.count - 1) where pts[k].count == 3 && pts[k + 1].count == 3 {
            let a = SIMD3(pts[k][0], pts[k][1], pts[k][2])
            let b = SIMD3(pts[k + 1][0], pts[k + 1][1], pts[k + 1][2])
            var dir = b - a; dir.y = 0
            let len = simd_length(dir)
            guard len > 1e-6 else { continue }
            dir /= len
            // Fully annotated: left to infer, the compound SIMD expression below
            // blows the type-checker's budget and fails to compile.
            let side = SIMD3<Double>(-dir.z, 0, dir.x)
            for offset in [-gauge, gauge] {
                let c: SIMD3<Double> = side * offset
                let w: SIMD3<Double> = side * railW
                let lift = SIMD3<Double>(0, 0.10, 0)
                let a0: SIMD3<Double> = a + c + lift
                let b0: SIMD3<Double> = b + c + lift
                let quad: [SIMD3<Double>] = [a0 - w, a0 + w, b0 - w, b0 + w]
                let idx = UInt32(vertices.count)
                for corner in quad {
                    vertices.append(Vertex(
                        position: SIMD3(Float(corner.x), Float(corner.y), Float(corner.z)),
                        normal: SIMD3(0, 1, 0), colour: steel))
                }
                indices += [idx, idx + 2, idx + 1, idx + 1, idx + 2, idx + 3]
            }
        }
    }

    // MARK: - Lamps

    private func addLamp(at base: SIMD3<Double>) {
        let h = 4.6, r = 0.075
        let metal = SIMD3<Float>(0.26, 0.27, 0.28)
        for i in 0..<4 {
            let a0 = Double(i) / 4 * 2 * .pi, a1 = Double(i + 1) / 4 * 2 * .pi
            let p0 = SIMD2(cos(a0) * r, sin(a0) * r)
            let p1 = SIMD2(cos(a1) * r, sin(a1) * r)
            let nrm = normalize(SIMD3<Float>(Float(p0.x + p1.x), 0, Float(p0.y + p1.y)))
            let idx = UInt32(vertices.count)
            for (p, y) in [(p0, 0.0), (p1, 0.0), (p0, h), (p1, h)] {
                vertices.append(Vertex(
                    position: SIMD3(Float(base.x + p.x), Float(base.y + y),
                                    Float(base.z + p.y)),
                    normal: nrm.x.isNaN ? SIMD3(0, 1, 0) : nrm, colour: metal))
            }
            indices += [idx, idx + 2, idx + 1, idx + 1, idx + 2, idx + 3]
        }
        // Lamp head.
        let idx = UInt32(vertices.count)
        let hw = 0.30
        for (dx, dz) in [(-hw, -hw), (hw, -hw), (-hw, hw), (hw, hw)] {
            vertices.append(Vertex(
                position: SIMD3(Float(base.x + dx), Float(base.y + h), Float(base.z + dz)),
                normal: SIMD3(0, 1, 0), colour: metal * 1.3))
        }
        indices += [idx, idx + 2, idx + 1, idx + 1, idx + 2, idx + 3]
    }

    // MARK: - Kerbs and pavement

    /// A road that simply stops and blends into the photograph reads as a
    /// painted stripe. A kerb gives the street an edge, and the raised pavement
    /// beside it samples the aerial photograph, so it keeps real paving detail.
    private func addPavements() {
        let kerbHeight = 0.14
        let width = 2.4
        let kerbColour = SIMD3<Float>(0.55, 0.54, 0.52)

        for edge in roadEdgesForPavement {
            // Service roads and alleys do not get formal pavements, and adding
            // them turns every courtyard into a maze of kerbs.
            let cls = edge.c.replacingOccurrences(of: "_link", with: "")
            guard ["primary", "secondary", "tertiary", "residential"].contains(cls) else {
                continue
            }
            let pts = edge.p.compactMap { p -> SIMD3<Double>? in
                p.count == 3 ? SIMD3(p[0], p[1], p[2]) : nil
            }
            guard pts.count >= 2 else { continue }
            let half = edge.w / 2

            for k in 0..<(pts.count - 1) {
                let a = pts[k], b = pts[k + 1]
                var dir = b - a; dir.y = 0
                let len = simd_length(dir)
                guard len > 1e-6 else { continue }
                dir /= len
                let side = SIMD3(-dir.z, 0, dir.x)

                for sign in [-1.0, 1.0] {
                    let inner = side * (half * sign)
                    let outer = side * ((half + width) * sign)
                    let low = SIMD3<Double>(0, 0.08, 0)
                    let high = SIMD3<Double>(0, kerbHeight, 0)

                    // Kerb face.
                    let kIdx = UInt32(vertices.count)
                    let nrm = SIMD3<Float>(Float(-side.x * sign), 0, Float(-side.z * sign))
                    for corner in [a + inner + low, b + inner + low,
                                   a + inner + high, b + inner + high] {
                        vertices.append(Vertex(
                            position: SIMD3(Float(corner.x), Float(corner.y), Float(corner.z)),
                            normal: nrm, colour: kerbColour))
                    }
                    indices += [kIdx, kIdx + 2, kIdx + 1, kIdx + 1, kIdx + 2, kIdx + 3]

                    // Pavement top, textured from the orthophoto.
                    let pIdx = UInt32(vertices.count)
                    for corner in [a + inner + high, a + outer + high,
                                   b + inner + high, b + outer + high] {
                        vertices.append(Vertex(
                            position: SIMD3(Float(corner.x), Float(corner.y), Float(corner.z)),
                            normal: SIMD3(0, 1, 0), colour: .one, material: .ortho))
                    }
                    indices += [pIdx, pIdx + 2, pIdx + 1, pIdx + 1, pIdx + 2, pIdx + 3]
                }
            }
        }
    }
}

private func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, t: Float) -> SIMD3<Float> {
    a + (b - a) * t
}

private func hash(_ n: Float) -> Float {
    let s = sin(n * 12.9898) * 43758.5453
    return s - s.rounded(.down)
}
