import Foundation
import simd

/// Find the world data whether we are run from the scratchpad or from the repo,
/// where the Swift package sits two levels below `data/`.
func locate(_ name: String) -> URL {
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let candidates = ["..", "../..", "../../data", "../data", "data", "."]
    for c in candidates {
        let url = cwd.appendingPathComponent(c).appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: url.path) { return url.standardized }
    }
    FileHandle.standardError.write(Data("could not find \(name)\n".utf8))
    exit(1)
}

let worldURL = locate("zurich_world.json")
let buildingsURL = locate("zurich_buildings.json")
let orthoImageURL = locate("zurich_ortho.jpg")
let orthoMetaURL = locate("zurich_ortho.json")
let here = worldURL.deletingLastPathComponent()

// The one mismatch that silently corrupts everything: keep Swift and Metal
// agreeing on the vertex stride.
precondition(MemoryLayout<Vertex>.stride == 64,
             "Vertex stride is \(MemoryLayout<Vertex>.stride), shader expects 64")

var clock = Date()
let mesh = try WorldMesh(worldURL: worldURL, buildingsURL: buildingsURL)
print(String(format: "Mesh built in %.2f s", -clock.timeIntervalSinceNow))
print("  \(mesh.vertices.count) vertices, \(mesh.indices.count / 3) triangles")
print("  terrain \(mesh.terrainRange.count / 3), roads \(mesh.roadRange.count / 3), "
    + "buildings \(mesh.buildingRange.count / 3) triangles")

clock = Date()
mesh.buildStreetscape(url: locate("zurich_streetscape.json"))
print(String(format: "Streetscape built in %.2f s — %d triangles",
             -clock.timeIntervalSinceNow, mesh.streetscapeRange.count / 3))
print("  total \(mesh.indices.count / 3) triangles")

let plateCount = mesh.buildPlates(url: locate("facade_atlas.json"))
print("Photographic façades: \(plateCount) plates")

let network = try RoadNetwork(contentsOf: worldURL)
let renderer = try Renderer(mesh: mesh)
clock = Date()
renderer.loadOrtho(imageURL: orthoImageURL, metaURL: orthoMetaURL)
print(String(format: "Ortho ground texture loaded in %.2f s", -clock.timeIntervalSinceNow))
clock = Date()
let materialsDir = worldURL.deletingLastPathComponent().appendingPathComponent("materials")
let panoCount = renderer.loadPanoramas(indexURL: locate("panoramas.json"))
print("Panoramas loaded: \(panoCount)")
if renderer.loadFacadeAtlas(url: locate("facade_atlas.jpg")) {
    print("Façade atlas loaded")
} else {
    print("Façade atlas NOT loaded")
}
if renderer.loadMaterials(directory: materialsDir) {
    print(String(format: "PBR materials loaded in %.2f s", -clock.timeIntervalSinceNow))
} else {
    print("PBR materials NOT loaded — falling back to procedural surfaces")
}

/// Eye point for a driver: on the road, at seat height, looking down the street.
func driverCamera(street: String, back: Float = 0, lift: Float = 1.35) -> Camera? {
    guard let spawn = network.spawn(on: street) else { return nil }
    let dir = SIMD3<Float>(Float(sin(spawn.yaw)), 0, Float(-cos(spawn.yaw)))
    let ground = Float(network.sampleGround(x: spawn.position.x, z: spawn.position.z).height)
    let eye = SIMD3<Float>(Float(spawn.position.x), ground + lift, Float(spawn.position.z))
              - dir * back
    return Camera(eye: eye, target: eye + dir * 60 + SIMD3(0, -0.06, 0), fovDegrees: 62)
}

if CommandLine.arguments.contains("drive") {
    print("Driving Zurich.  W/S throttle & brake · A/D steer · R respawn · Esc quit")
    runInteractive(mesh: mesh, network: network, renderer: renderer)
    exit(0)
}

struct Shot {
    let name: String
    let camera: Camera
    let note: String
}

var shots: [Shot] = []

for street in ["Freiestrasse", "Hottingerstrasse", "Zeltweg"] {
    if let cam = driverCamera(street: street) {
        shots.append(Shot(name: "street-\(street.lowercased())", camera: cam,
                          note: "driver's eye on \(street)"))
    }
}

// An oblique over the middle of the city, high enough to read the street plan.
shots.append(Shot(
    name: "aerial",
    camera: Camera(eye: SIMD3(900, 430, 1150), target: SIMD3(1500, 40, 500), fovDegrees: 48),
    note: "oblique over Hottingen"))

// Low over the Limmat, where the bridges and the old town are.
shots.append(Shot(
    name: "hottingen-low",
    camera: Camera(eye: SIMD3(1280, 55, 820), target: SIMD3(1560, 25, 430), fovDegrees: 55),
    note: "low over Freiestrasse"))

shots.append(Shot(
    name: "pano",
    camera: Camera(eye: SIMD3(1633.2, 38.4, 609.6),
                   target: SIMD3(1593.7, 36.9, 615.9),
                   fovDegrees: 70),
    note: "projective panorama"))

let width = 1400, height = 850
for shot in shots {
    clock = Date()
    let image = try renderer.render(camera: shot.camera, width: width, height: height)
    let out = here.appendingPathComponent("shot-\(shot.name).png")
    try Renderer.write(image, to: out)
    print(String(format: "  %-28@ %.0f ms  %@", shot.name,
                 -clock.timeIntervalSinceNow * 1000, shot.note))
}
print("done")
