// Minimal stand-ins for the ApexSim API that RoadNetwork depends on, so this
// compiles and runs standalone without touching the apexline tree.
import simd
public typealias Vec3 = SIMD3<Double>
public enum Sim { public static let up = Vec3(0, 1, 0) }
public enum SurfaceKind: Int, Sendable, CaseIterable, Codable {
    case asphalt, kerb, grass, gravel
    public var gripMultiplier: Double {
        switch self { case .asphalt: 1.0; case .kerb: 0.92; case .grass: 0.45; case .gravel: 0.55 }
    }
}
public struct GroundSample: Sendable {
    public var height: Double; public var normal: Vec3; public var surface: SurfaceKind
    public init(height: Double, normal: Vec3 = Sim.up, surface: SurfaceKind = .asphalt) {
        self.height = height; self.normal = normal; self.surface = surface
    }
}
public protocol GroundProvider: AnyObject {
    func sampleGround(x: Double, z: Double) -> GroundSample
}
