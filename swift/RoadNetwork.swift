import Foundation
import simd

/// An open-world street network you can drive anywhere in, as opposed to a
/// closed circuit you lap.
///
/// This is the counterpart to `Track`: both answer the one question the physics
/// ever asks the world — `sampleGround(x:z:)` — so `VehicleSim`, `TireModel` and
/// the suspension work here completely unchanged. What does *not* apply is
/// everything that assumes a lap: `RacingLine`, `Ghost` and lap timing all need
/// a closed centreline, and a city has no such thing.
///
/// The data is generated from OpenStreetMap centrelines and swisstopo's
/// swissALTI3D terrain by `Tools/zurich_world.py`. Roads are baked onto the
/// terrain at export time, so a road and the ground under it agree by
/// construction rather than by luck.
///
/// Road data © OpenStreetMap contributors (ODbL). Terrain © swisstopo.
public final class RoadNetwork: GroundProvider {

    // MARK: - Stored world

    public struct Road: Sendable {
        public var points: [Vec3]
        public var width: Double
        public var name: String
        public var roadClass: String
        public var speedLimit: Int
        public var isOneway: Bool
        public var isBridge: Bool
        public var isTunnel: Bool
    }

    public private(set) var roads: [Road] = []
    public let name: String

    /// Half-width band outside the carriageway that reads as kerb before the
    /// ground turns to whatever is beyond the road. Keeps a wheel dropping off
    /// the edge from stepping discontinuously, which would launch the car.
    private let kerbBand: Double = 0.9

    // Terrain grid.
    private let tx0: Double, tz0: Double, tcell: Double
    private let tnx: Int, tnz: Int
    private let heights: [Double]

    // Flattened road segments plus a uniform spatial hash over XZ, so a ground
    // query touches a handful of segments instead of all quarter-million.
    private struct Segment {
        var a: Vec3
        var b: Vec3
        var halfWidth: Double
        var road: Int
    }
    private var segments: [Segment] = []
    private var grid: [Int64: [Int32]] = [:]
    private let cellSize: Double = 40

    // MARK: - Loading

    public init(data: Data) throws {
        let world = try JSONDecoder().decode(WorldFile.self, from: data)
        self.name = world.name
        self.tx0 = world.terrain.x0
        self.tz0 = world.terrain.z0
        self.tcell = world.terrain.cell
        self.tnx = world.terrain.nx
        self.tnz = world.terrain.nz
        self.heights = world.terrain.heights

        roads.reserveCapacity(world.edges.count)
        for (i, edge) in world.edges.enumerated() {
            let pts = edge.p.compactMap { p -> Vec3? in
                p.count == 3 ? Vec3(p[0], p[1], p[2]) : nil
            }
            guard pts.count >= 2 else { continue }
            roads.append(Road(points: pts, width: edge.w, name: edge.n,
                              roadClass: edge.c, speedLimit: edge.s,
                              isOneway: edge.o, isBridge: edge.b, isTunnel: edge.t))
            for k in 0..<(pts.count - 1) {
                segments.append(Segment(a: pts[k], b: pts[k + 1],
                                        halfWidth: edge.w / 2, road: i))
            }
        }
        buildGrid()
    }

    public convenience init(contentsOf url: URL) throws {
        try self.init(data: Data(contentsOf: url))
    }

    private func buildGrid() {
        for (i, seg) in segments.enumerated() {
            let pad = seg.halfWidth + kerbBand
            let minX = min(seg.a.x, seg.b.x) - pad, maxX = max(seg.a.x, seg.b.x) + pad
            let minZ = min(seg.a.z, seg.b.z) - pad, maxZ = max(seg.a.z, seg.b.z) + pad
            for cx in Int(floor(minX / cellSize))...Int(floor(maxX / cellSize)) {
                for cz in Int(floor(minZ / cellSize))...Int(floor(maxZ / cellSize)) {
                    grid[key(cx, cz), default: []].append(Int32(i))
                }
            }
        }
    }

    private func key(_ x: Int, _ z: Int) -> Int64 {
        (Int64(x) << 32) ^ Int64(UInt32(bitPattern: Int32(z)))
    }

    // MARK: - GroundProvider

    /// - Note: the protocol hands us only an XZ position, with no height, so a
    ///   flyover and the street beneath it are indistinguishable here — we take
    ///   whichever centreline is nearer. Zurich's 92 bridges are almost all over
    ///   water or rail rather than over other drivable road, so this is close to
    ///   free in practice. Genuine stacked roads would need the protocol to pass
    ///   the wheel's height too.
    public func sampleGround(x: Double, z: Double) -> GroundSample {
        let cx = Int(floor(x / cellSize)), cz = Int(floor(z / cellSize))
        var best: (distance: Double, height: Double, normal: Vec3, halfWidth: Double)?

        if let bucket = grid[key(cx, cz)] {
            for idx in bucket {
                let seg = segments[Int(idx)]
                let (d, y, n) = distanceToSegment(seg, x: x, z: z)
                guard d <= seg.halfWidth + kerbBand else { continue }
                // Rank by how far outside this road's own edge we are, not by raw
                // distance: a wide boulevard 5 m from its centreline is still road,
                // while a narrow lane 5 m out is not, and raw distance would pick
                // the wrong one of the two.
                let overhang = d - seg.halfWidth
                if overhang < (best.map { $0.distance - $0.halfWidth } ?? .infinity) {
                    best = (d, y, n, seg.halfWidth)
                }
            }
        }

        let groundY = terrainHeight(x: x, z: z)
        guard let hit = best else {
            return GroundSample(height: groundY, normal: terrainNormal(x: x, z: z),
                                surface: .grass)
        }

        // On the carriageway the road wins outright. Across the kerb band we ease
        // from the road surface down to the terrain, so a wheel dropping off the
        // edge meets a ramp rather than a step that would launch the car.
        if hit.distance <= hit.halfWidth {
            return GroundSample(height: hit.height, normal: hit.normal, surface: .asphalt)
        }
        let t = max(0, min(1, (hit.distance - hit.halfWidth) / kerbBand))
        return GroundSample(height: hit.height * (1 - t) + groundY * t,
                            normal: hit.normal, surface: t < 0.5 ? .kerb : .grass)
    }

    /// Perpendicular distance in XZ to a segment, the interpolated road height
    /// at the closest point, and the road's surface normal.
    private func distanceToSegment(_ seg: Segment, x: Double, z: Double)
        -> (Double, Double, Vec3) {
        let ax = seg.a.x, az = seg.a.z
        let dx = seg.b.x - ax, dz = seg.b.z - az
        let lenSq = dx * dx + dz * dz
        var t = 0.0
        if lenSq > 1e-9 {
            t = max(0, min(1, ((x - ax) * dx + (z - az) * dz) / lenSq))
        }
        let px = ax + dx * t, pz = az + dz * t
        let distance = (Vec3(x, 0, z) - Vec3(px, 0, pz)).length
        let height = seg.a.y + (seg.b.y - seg.a.y) * t

        let tangent = (seg.b - seg.a).normalizedOrZero
        let right = cross(Sim.up, tangent).normalizedOrZero
        let normal = right == .zero ? Sim.up : cross(tangent, right).normalizedOrZero
        return (distance, height, normal == .zero ? Sim.up : normal)
    }

    // MARK: - Terrain

    private func terrainHeight(x: Double, z: Double) -> Double {
        let fx = (x - tx0) / tcell, fz = (z - tz0) / tcell
        let i = max(0, min(tnx - 2, Int(floor(fx))))
        let j = max(0, min(tnz - 2, Int(floor(fz))))
        let sx = max(0, min(1, fx - Double(i))), sz = max(0, min(1, fz - Double(j)))
        let h00 = heights[j * tnx + i], h10 = heights[j * tnx + i + 1]
        let h01 = heights[(j + 1) * tnx + i], h11 = heights[(j + 1) * tnx + i + 1]
        return h00 * (1 - sx) * (1 - sz) + h10 * sx * (1 - sz)
             + h01 * (1 - sx) * sz + h11 * sx * sz
    }

    private func terrainNormal(x: Double, z: Double) -> Vec3 {
        let d = tcell * 0.5
        let dhdx = (terrainHeight(x: x + d, z: z) - terrainHeight(x: x - d, z: z)) / (2 * d)
        let dhdz = (terrainHeight(x: x, z: z + d) - terrainHeight(x: x, z: z - d)) / (2 * d)
        return Vec3(-dhdx, 1, -dhdz).normalizedOrZero
    }

    // MARK: - Finding somewhere to start

    /// A point on a named street, facing along it. Free roam has no grid slot to
    /// drop the player into, so spawning is by street name instead.
    public func spawn(on street: String) -> (position: Vec3, yaw: Double)? {
        guard let road = roads.first(where: {
            $0.name.compare(street, options: .caseInsensitive) == .orderedSame
        }), road.points.count >= 2 else { return nil }
        let mid = road.points.count / 2
        let a = road.points[max(0, mid - 1)], b = road.points[min(road.points.count - 1, mid)]
        let dir = (b - a).normalizedOrZero
        return (a, atan2(dir.x, -dir.z))
    }

    public var streetNames: [String] {
        Array(Set(roads.map(\.name).filter { !$0.isEmpty })).sorted()
    }
}

// MARK: - On-disk format

private struct WorldFile: Decodable {
    struct Terrain: Decodable {
        let x0: Double, z0: Double, cell: Double
        let nx: Int, nz: Int
        let heights: [Double]
    }
    struct Edge: Decodable {
        let p: [[Double]]
        let w: Double
        let n: String
        let c: String
        let s: Int
        let o: Bool
        let b: Bool
        let t: Bool
    }
    let name: String
    let terrain: Terrain
    let edges: [Edge]
}

private extension Vec3 {
    var length: Double { (x * x + y * y + z * z).squareRoot() }
    var normalizedOrZero: Vec3 {
        let l = length
        return l > 1e-9 ? self / l : .zero
    }
}
