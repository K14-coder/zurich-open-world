import Foundation
import simd

/// The carriageway surface.
///
/// Three things have to be right before a road network extruded from OSM
/// centrelines stops looking generated. All three are about what happens where
/// two pieces of road meet:
///
///  * **Mitred joins.** Offsetting each segment along its own perpendicular
///    leaves a wedge-shaped notch on the outside of every bend and a self-overlap
///    on the inside — the sharper the corner, the worse it gets. Sharing a single
///    offset vertex between neighbouring segments, placed along the angle
///    bisector, makes the ribbon continuous.
///  * **Trimmed approaches.** Every edge runs to the *centre* of its junction, so
///    at a crossroads four carriageways lie on top of one another at the same
///    height and z-fight into a shimmering mess. Pulling each approach back to
///    the edge of the junction leaves a clean hole instead.
///  * **Junction fill.** That hole is then covered by one polygon built from the
///    trimmed ends, which is what an intersection actually looks like from above.
///
/// Without these the mesh reads as bad *map data*, which is the usual diagnosis
/// and almost always the wrong one — the centrelines are fine, it's the topology
/// of the ribbon that's broken.
extension WorldMesh {

    /// Lift above the terrain, m. The road heights are sampled from the same grid
    /// as the ground mesh, so without this they are coplanar and shimmer.
    private static var surfaceLift: Double { 0.08 }
    /// Junctions sit a hair above the ribbons, so wherever the two still overlap
    /// the junction wins every time rather than fighting per-pixel.
    private static var junctionLift: Double { 0.095 }

    func buildRoadSurface(_ edges: [WorldJSON.Edge]) {
        let roads = polylines(from: edges)
        guard !roads.isEmpty else { return }

        // Endpoints were cut at shared OSM nodes by the Python pipeline and
        // rounded to the centimetre, so matching them recovers the junction graph
        // exactly — no spatial tolerance guesswork needed.
        var arms: [Int64: [JunctionArm]] = [:]
        for (index, road) in roads.enumerated() {
            guard let head = road.points.first, let tail = road.points.last else { continue }
            arms[Self.junctionKey(head), default: []].append(
                JunctionArm(road: index, atStart: true, half: road.half, colour: road.colour))
            arms[Self.junctionKey(tail), default: []].append(
                JunctionArm(road: index, atStart: false, half: road.half, colour: road.colour))
        }

        // How far back each approach is pulled. The widest road at the junction
        // sets the radius: anything smaller would leave the widest one still
        // crossing the middle.
        var radius: [Int64: Double] = [:]
        for (key, list) in arms where list.count >= 3 {
            let widest = list.map(\.half).max() ?? 0
            radius[key] = min(16, widest * 1.15 + 0.4)
        }

        var trimmed = roads
        for index in roads.indices {
            guard let head = roads[index].points.first,
                  let tail = roads[index].points.last else { continue }
            trimmed[index].points = Self.trim(
                roads[index].points,
                fromStart: radius[Self.junctionKey(head)] ?? 0,
                fromEnd: radius[Self.junctionKey(tail)] ?? 0
            )
        }

        for road in trimmed { addRibbon(road) }
        for (_, list) in arms where list.count >= 3 { addJunction(arms: list, roads: trimmed) }
    }

    // MARK: - Input

    private func polylines(from edges: [WorldJSON.Edge]) -> [RoadPolyline] {
        var roads: [RoadPolyline] = []
        roads.reserveCapacity(edges.count)
        for edge in edges {
            var points: [SIMD3<Double>] = []
            for p in edge.p where p.count == 3 {
                let v = SIMD3(p[0], p[1], p[2])
                // Repeated points are common in OSM ways and make every direction
                // downstream of them undefined.
                if let last = points.last, Self.planarDistance(last, v) < 1e-4 { continue }
                points.append(v)
            }
            guard points.count >= 2 else { continue }
            roads.append(RoadPolyline(points: points,
                                      half: max(edge.w, 1.0) / 2,
                                      colour: roadColour(edge.c)))
        }
        return roads
    }

    // MARK: - Ribbon

    private func addRibbon(_ road: RoadPolyline) {
        let points = road.points
        guard points.count >= 2 else { return }
        let lift = Self.surfaceLift

        var left: [SIMD3<Double>] = [], right: [SIMD3<Double>] = []
        left.reserveCapacity(points.count)
        right.reserveCapacity(points.count)
        for i in points.indices {
            let offset = Self.mitre(points, i) * road.half
            let p = SIMD3(points[i].x, points[i].y + lift, points[i].z)
            left.append(SIMD3(p.x + offset.x, p.y, p.z + offset.y))
            right.append(SIMD3(p.x - offset.x, p.y, p.z - offset.y))
        }

        for i in 0..<(points.count - 1) {
            let base = UInt32(vertices.count)
            for corner in [right[i], left[i], right[i + 1], left[i + 1]] {
                vertices.append(Vertex(
                    position: SIMD3(Float(corner.x), Float(corner.y), Float(corner.z)),
                    normal: SIMD3(0, 1, 0), colour: road.colour, material: .road))
            }
            indices += [base, base + 2, base + 1, base + 1, base + 2, base + 3]
        }
    }

    /// Offset direction at vertex `i`, scaled so both carriageway edges stay
    /// exactly half a width from the centreline through the corner.
    private static func mitre(_ points: [SIMD3<Double>], _ i: Int) -> SIMD2<Double> {
        let before = i > 0 ? normal(points[i - 1], points[i]) : nil
        let after = i < points.count - 1 ? normal(points[i], points[i + 1]) : nil

        switch (before, after) {
        case let (.some(b), .some(a)):
            var bisector = b + a
            let length = simd_length(bisector)
            // A way that doubles straight back on itself has no bisector.
            guard length > 1e-6 else { return a }
            bisector /= length
            // 1/cos(θ/2) is exact, but goes to infinity as the corner closes, so
            // it is capped — a hairpin gets a slightly pinched join rather than a
            // spike shooting off across the city.
            let cosHalf = simd_dot(bisector, a)
            return bisector * min(3.0, 1.0 / max(cosHalf, 0.34))
        case let (.some(b), .none): return b
        case let (.none, .some(a)): return a
        case (.none, .none): return SIMD2(0, 0)
        }
    }

    private static func normal(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> SIMD2<Double>? {
        var direction = SIMD2(b.x - a.x, b.z - a.z)
        let length = simd_length(direction)
        guard length > 1e-9 else { return nil }
        direction /= length
        return SIMD2(-direction.y, direction.x)
    }

    // MARK: - Junctions

    private func addJunction(arms: [JunctionArm], roads: [RoadPolyline]) {
        /// One approach's mouth: where it was trimmed to, and its two corners.
        struct Mouth {
            var bearing: Double
            var clockwise: SIMD3<Double>
            var counter: SIMD3<Double>
            var half: Double
            var colour: SIMD3<Float>
        }

        var mouths: [Mouth] = []
        for arm in arms {
            let points = roads[arm.road].points
            guard points.count >= 2 else { continue }
            // The trimmed end facing this junction, and the direction it runs away in.
            let end = arm.atStart ? points[0] : points[points.count - 1]
            let next = arm.atStart ? points[1] : points[points.count - 2]
            let away = SIMD2(next.x - end.x, next.z - end.z)
            guard simd_length(away) > 1e-9, let n = Self.normal(end, next) else { continue }

            mouths.append(Mouth(
                bearing: atan2(away.y, away.x),
                clockwise: SIMD3(end.x - n.x * arm.half, end.y, end.z - n.y * arm.half),
                counter: SIMD3(end.x + n.x * arm.half, end.y, end.z + n.y * arm.half),
                half: arm.half, colour: arm.colour
            ))
        }
        guard mouths.count >= 3 else { return }

        // Order the *approaches* by bearing and take each one's two corners
        // together, rather than sorting all the corners as one soup. Sorting
        // corners individually lets a wide road's mouth swallow the corners of a
        // narrow neighbour, and the ring comes out as a self-intersecting star.
        mouths.sort { $0.bearing < $1.bearing }
        var ring: [SIMD3<Double>] = []
        ring.reserveCapacity(mouths.count * 2)
        for mouth in mouths {
            ring.append(mouth.clockwise)
            ring.append(mouth.counter)
        }

        let widest = mouths.max { $0.half < $1.half }
        let colour = widest?.colour ?? SIMD3<Float>(0.3, 0.3, 0.3)

        let count = Double(ring.count)
        let centre = SIMD3(
            ring.reduce(0) { $0 + $1.x } / count,
            ring.reduce(0) { $0 + $1.y } / count,
            ring.reduce(0) { $0 + $1.z } / count
        )

        let base = UInt32(vertices.count)
        vertices.append(Vertex(
            position: SIMD3(Float(centre.x), Float(centre.y + Self.junctionLift), Float(centre.z)),
            normal: SIMD3(0, 1, 0), colour: colour, material: .road))
        for corner in ring {
            vertices.append(Vertex(
                position: SIMD3(Float(corner.x), Float(corner.y + Self.junctionLift), Float(corner.z)),
                normal: SIMD3(0, 1, 0), colour: colour, material: .road))
        }

        for i in 0..<ring.count {
            let j = (i + 1) % ring.count
            let a = ring[i], b = ring[j]
            guard Self.planarDistance(a, b) > 1e-6 else { continue }
            let v1 = SIMD2(a.x - centre.x, a.z - centre.z)
            let v2 = SIMD2(b.x - centre.x, b.z - centre.z)
            let upwards = (v1.y * v2.x - v1.x * v2.y) > 0
            let i0 = base + 1 + UInt32(i)
            let i1 = base + 1 + UInt32(j)
            indices += upwards ? [base, i0, i1] : [base, i1, i0]
        }
    }

    // MARK: - Trimming

    private static func trim(_ points: [SIMD3<Double>],
                             fromStart head: Double, fromEnd tail: Double) -> [SIMD3<Double>] {
        var head = head, tail = tail
        guard head > 0.01 || tail > 0.01 else { return points }

        let total = planarLength(points)
        // Always leave a metre of carriageway. Short links between two big
        // junctions would otherwise be trimmed out of existence from both ends.
        let keep = 1.0
        guard total > keep else { return points }
        if head + tail > total - keep {
            let scale = (total - keep) / (head + tail)
            head *= scale
            tail *= scale
        }

        var result = points
        if head > 0.01 { result = advance(result, by: head) }
        if tail > 0.01 {
            result = Array(advance(Array(result.reversed()), by: tail).reversed())
        }
        return result
    }

    /// Drops the first `distance` metres of a polyline, splitting the segment it
    /// lands in.
    private static func advance(_ points: [SIMD3<Double>], by distance: Double) -> [SIMD3<Double>] {
        var remaining = distance
        for i in 0..<(points.count - 1) {
            let a = points[i], b = points[i + 1]
            let segment = planarDistance(a, b)
            guard segment > 1e-9 else { continue }
            if remaining >= segment {
                remaining -= segment
                continue
            }
            let t = remaining / segment
            var result = [a + (b - a) * t]
            result.append(contentsOf: points[(i + 1)...])
            return result
        }
        return Array(points.suffix(2))
    }

    // MARK: - Small helpers

    private static func planarDistance(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
        simd_length(SIMD2(b.x - a.x, b.z - a.z))
    }

    private static func planarLength(_ points: [SIMD3<Double>]) -> Double {
        var total = 0.0
        for i in 0..<(points.count - 1) { total += planarDistance(points[i], points[i + 1]) }
        return total
    }

    /// Pull-back radius per junction, against the same graph the carriageway
    /// uses, so layers drawn on top of the roads can be trimmed identically.
    ///
    /// The pavement needs this as much as the carriageway does. Generated
    /// per-edge with no junction awareness it walks straight across every
    /// intersection, and because it sits above the tarmac it covers exactly the
    /// corners the junction fill was added to get right.
    ///
    /// `extra` widens the radius for layers that sit outside the carriageway:
    /// a kerb line 2.4 m out reaches further into the junction than the tarmac
    /// does, so it has to be pulled back further.
    func junctionTrimRadii(_ edges: [WorldJSON.Edge], extra: Double = 0) -> [Int64: Double] {
        let roads = polylines(from: edges)
        var widest: [Int64: Double] = [:]
        var count: [Int64: Int] = [:]
        for road in roads {
            guard let head = road.points.first, let tail = road.points.last else { continue }
            for key in [Self.junctionKey(head), Self.junctionKey(tail)] {
                widest[key] = max(widest[key] ?? 0, road.half)
                count[key, default: 0] += 1
            }
        }
        var radii: [Int64: Double] = [:]
        for (key, n) in count where n >= 3 {
            radii[key] = min(16 + extra, (widest[key] ?? 0) * 1.15 + 0.4 + extra)
        }
        return radii
    }

    /// Trim a polyline back from whichever of its ends sit at a junction.
    static func trimAtJunctions(_ points: [SIMD3<Double>],
                                radii: [Int64: Double]) -> [SIMD3<Double>] {
        guard let head = points.first, let tail = points.last else { return points }
        return trim(points,
                    fromStart: radii[junctionKey(head)] ?? 0,
                    fromEnd: radii[junctionKey(tail)] ?? 0)
    }

    /// Endpoints quantised to the centimetre, packed into one integer so junctions
    /// can be looked up in a dictionary.
    fileprivate static func junctionKey(_ p: SIMD3<Double>) -> Int64 {
        let x = Int64((p.x * 100).rounded())
        let z = Int64((p.z * 100).rounded())
        return x &* 73_856_093 &+ z &* 19_349_663
    }
}

private struct RoadPolyline {
    var points: [SIMD3<Double>]
    var half: Double
    var colour: SIMD3<Float>
}

private struct JunctionArm {
    var road: Int
    var atStart: Bool
    var half: Double
    var colour: SIMD3<Float>
}
