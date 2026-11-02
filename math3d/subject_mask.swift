// SUBJECT MASK: Apple Vision lifts the subject under a given point out of an image (foreground instance mask), on
// this Mac; nothing is sent anywhere.  Output: a greyscale PNG the image's size, white on the subject.
//     swiftc -O subject_mask.swift -o subject_mask && ./subject_mask in.png out.png X Y   (X, Y in pixels, y down)
import AppKit
import CoreImage
import Foundation
import Vision

let args = CommandLine.arguments
guard args.count >= 5, let img = CIImage(contentsOf: URL(fileURLWithPath: args[1])),
      let px = Double(args[3]), let py = Double(args[4]) else {
    FileHandle.standardError.write("usage: subject_mask in.png out.png X Y\n".data(using: .utf8)!)
    exit(1)
}
let handler = VNImageRequestHandler(ciImage: img, options: [:])
let req = VNGenerateForegroundInstanceMaskRequest()
do { try handler.perform([req]) } catch { FileHandle.standardError.write("vision failed\n".data(using: .utf8)!); exit(2) }
guard let obs = req.results?.first else { FileHandle.standardError.write("no subject\n".data(using: .utf8)!); exit(3) }
let labels = obs.instanceMask
CVPixelBufferLockBaseAddress(labels, .readOnly)
let lw = CVPixelBufferGetWidth(labels), lh = CVPixelBufferGetHeight(labels)
let row = CVPixelBufferGetBytesPerRow(labels)
let base = CVPixelBufferGetBaseAddress(labels)!.assumingMemoryBound(to: UInt8.self)
let W = Double(img.extent.width), H = Double(img.extent.height)
let lx = min(lw - 1, max(0, Int(px / W * Double(lw)))), ly = min(lh - 1, max(0, Int(py / H * Double(lh))))
var label = Int(base[ly * row + lx])
if label == 0 {                                              // nearest labelled pixel to the point
    var best = Double.infinity
    for y in 0..<lh { for x in 0..<lw where base[y * row + x] != 0 {
        let d = Double((x - lx) * (x - lx) + (y - ly) * (y - ly))
        if d < best { best = d; label = Int(base[y * row + x]) }
    } }
}
CVPixelBufferUnlockBaseAddress(labels, .readOnly)
guard label != 0, let mask = try? obs.generateScaledMaskForImage(forInstances: IndexSet(integer: label), from: handler)
else { FileHandle.standardError.write("no instance\n".data(using: .utf8)!); exit(4) }
let ci = CIImage(cvPixelBuffer: mask)
let ctx = CIContext()
guard let cg = ctx.createCGImage(ci, from: ci.extent) else { exit(5) }
let rep = NSBitmapImageRep(cgImage: cg)
guard let png = rep.representation(using: .png, properties: [:]) else { exit(6) }
try png.write(to: URL(fileURLWithPath: args[2]))
print("instance \(label) of \(obs.allInstances.count)")
