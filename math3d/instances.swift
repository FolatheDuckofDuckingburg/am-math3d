// INSTANCES: Apple Vision lifts every foreground subject out of an image (foreground instance masks), on this Mac;
// nothing is sent anywhere.  Output: a greyscale PNG the image's size whose grey level is the instance's number
// (0 = background), and one JSON line per instance: its number, its box (x0, y0, x1, y1 in pixels, y down), its area.
//     swiftc -O instances.swift -o instances && ./instances in.png labels.png
import AppKit
import CoreImage
import Foundation
import Vision

let args = CommandLine.arguments
guard args.count >= 3, let img = CIImage(contentsOf: URL(fileURLWithPath: args[1])) else {
    FileHandle.standardError.write("usage: instances in.png labels.png\n".data(using: .utf8)!)
    exit(1)
}
let W = Int(img.extent.width), H = Int(img.extent.height)
var labels = [UInt8](repeating: 0, count: W * H)
var lines: [String] = []
let handler = VNImageRequestHandler(ciImage: img, options: [:])
let req = VNGenerateForegroundInstanceMaskRequest()
do { try handler.perform([req]) } catch { FileHandle.standardError.write("vision failed\n".data(using: .utf8)!); exit(2) }
if let obs = req.results?.first {
    for inst in obs.allInstances.sorted() {
        guard let mask = try? obs.generateScaledMaskForImage(forInstances: IndexSet(integer: inst), from: handler) else { continue }
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        let mw = CVPixelBufferGetWidth(mask), mh = CVPixelBufferGetHeight(mask), row = CVPixelBufferGetBytesPerRow(mask)
        let fmt = CVPixelBufferGetPixelFormatType(mask)
        let base = CVPixelBufferGetBaseAddress(mask)!
        var x0 = W, y0 = H, x1 = -1, y1 = -1, area = 0
        for y in 0..<min(mh, H) {
            for x in 0..<min(mw, W) {
                let on: Bool
                if fmt == kCVPixelFormatType_OneComponent32Float {
                    on = base.advanced(by: y * row).assumingMemoryBound(to: Float32.self)[x] > 0.5
                } else {
                    on = base.advanced(by: y * row).assumingMemoryBound(to: UInt8.self)[x] > 127
                }
                if on {
                    labels[y * W + x] = UInt8(min(255, inst))
                    area += 1
                    x0 = min(x0, x); y0 = min(y0, y); x1 = max(x1, x); y1 = max(y1, y)
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(mask, .readOnly)
        if area > 0 { lines.append("{\"instance\": \(inst), \"box\": [\(x0), \(y0), \(x1), \(y1)], \"area\": \(area)}") }
    }
}
let cs = CGColorSpaceCreateDeviceGray()
let data = Data(labels) as CFData
guard let provider = CGDataProvider(data: data),
      let cg = CGImage(width: W, height: H, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: W, space: cs,
                       bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent),
      let png = NSBitmapImageRep(cgImage: cg).representation(using: .png, properties: [:]) else { exit(5) }
try png.write(to: URL(fileURLWithPath: args[2]))
lines.forEach { print($0) }
