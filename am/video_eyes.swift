// VIDEO EYES: Apple Vision on this Mac, nothing sent anywhere.  For every frame of a video: the salient objects
// (objectness saliency), people and animals are detected every DETECT frames; each new object (not overlapping a live
// track) starts a tracker (VNTrackObjectRequest), and every live track is followed frame to frame until its confidence
// falls below MINCONF.  Output: one JSON line per frame, {"t": frame, "tracks": [[id, x, y, w, h, conf], ...]}, boxes in
// pixels, x and y the box centre, y down.
//     swiftc -O video_eyes.swift -o video_eyes && ./video_eyes clip.mp4 > clip.tracks.jsonl
import AVFoundation
import Foundation
import Vision

let DETECT = 10
let MINCONF: Float = 0.3
let MAXTRACKS = 6

func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
    let i = a.intersection(b)
    if i.isNull { return 0 }
    let inter = i.width * i.height
    return inter / (a.width * a.height + b.width * b.height - inter)
}

let path = CommandLine.arguments[1]
let asset = AVURLAsset(url: URL(fileURLWithPath: path))
let sem = DispatchSemaphore(value: 0)
var videoTrack: AVAssetTrack?
Task {
    videoTrack = try? await asset.loadTracks(withMediaType: .video).first
    sem.signal()
}
sem.wait()
guard let vt = videoTrack, let reader = try? AVAssetReader(asset: asset) else {
    FileHandle.standardError.write("cannot read \(path)\n".data(using: .utf8)!)
    exit(1)
}
let out = AVAssetReaderTrackOutput(track: vt, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
reader.add(out)
reader.startReading()

let seq = VNSequenceRequestHandler()
var live: [Int: VNDetectedObjectObservation] = [:]
var nextId = 1
var t = 0
while let sample = out.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sample) {
    let W = CGFloat(CVPixelBufferGetWidth(pb)), H = CGFloat(CVPixelBufferGetHeight(pb))
    var rows: [[Double]] = []
    // follow the live tracks
    if !live.isEmpty {
        let reqs = live.map { (id, obs) -> (Int, VNTrackObjectRequest) in
            let r = VNTrackObjectRequest(detectedObjectObservation: obs)
            r.trackingLevel = .accurate
            return (id, r)
        }
        try? seq.perform(reqs.map { $0.1 }, on: pb, orientation: .up)
        for (id, r) in reqs {
            if let o = r.results?.first as? VNDetectedObjectObservation, o.confidence >= MINCONF {
                live[id] = o
            } else {
                live[id] = nil
            }
        }
    }
    // look for new objects
    if t % DETECT == 0 && live.count < MAXTRACKS {
        let h = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up)
        let sal = VNGenerateObjectnessBasedSaliencyImageRequest()
        let hum = VNDetectHumanRectanglesRequest()
        let ani = VNRecognizeAnimalsRequest()
        try? h.perform([sal, hum, ani])
        var boxes: [CGRect] = []
        if let s = sal.results?.first as? VNSaliencyImageObservation {
            boxes += (s.salientObjects ?? []).map { $0.boundingBox }
        }
        boxes += (hum.results ?? []).map { $0.boundingBox }
        boxes += (ani.results ?? []).map { $0.boundingBox }
        for b in boxes where b.width > 0.03 && b.height > 0.03 && b.width < 0.9 && b.height < 0.9 {
            if live.count >= MAXTRACKS { break }
            if live.values.contains(where: { iou($0.boundingBox, b) > 0.3 }) { continue }
            live[nextId] = VNDetectedObjectObservation(boundingBox: b)
            nextId += 1
        }
    }
    for (id, o) in live.sorted(by: { $0.key < $1.key }) {
        let b = o.boundingBox
        rows.append([Double(id), Double((b.midX) * W), Double((1 - b.midY) * H), Double(b.width * W), Double(b.height * H),
                     Double(o.confidence)])
    }
    let line: [String: Any] = ["t": t, "tracks": rows]
    if let d = try? JSONSerialization.data(withJSONObject: line), let s = String(data: d, encoding: .utf8) {
        print(s)
    }
    t += 1
}
