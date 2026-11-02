// TRACK FROM: Apple Vision follows given objects through a video, starting from given boxes at frame 0; nothing is
// sent anywhere.  Each box starts a tracker (VNTrackObjectRequest, accurate), as in video_eyes; a track ends when
// its confidence falls below MINCONF.  No new objects are looked for.  Boxes in and out are in pixels of the video
// itself, x and y the box centre, y down.  Output as video_eyes: one JSON line per frame,
// {"t": frame, "tracks": [[id, x, y, w, h, conf], ...]}.
//     swiftc -O track_from.swift -o track_from && ./track_from clip.mp4 '[[id, x, y, w, h], ...]'
import AVFoundation
import Foundation
import Vision

let MINCONF: Float = 0.3

let path = CommandLine.arguments[1]
guard let given = try? JSONSerialization.jsonObject(with: CommandLine.arguments[2].data(using: .utf8)!) as? [[Double]]
else {
    FileHandle.standardError.write("boxes: a JSON list of [id, x, y, w, h]\n".data(using: .utf8)!)
    exit(1)
}
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
var t = 0
while let sample = out.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sample) {
    let W = CGFloat(CVPixelBufferGetWidth(pb)), H = CGFloat(CVPixelBufferGetHeight(pb))
    if t == 0 {
        for g in given where g.count >= 5 {
            let r = CGRect(x: (CGFloat(g[1]) - CGFloat(g[3]) / 2) / W, y: 1 - (CGFloat(g[2]) + CGFloat(g[4]) / 2) / H,
                           width: CGFloat(g[3]) / W, height: CGFloat(g[4]) / H)
            live[Int(g[0])] = VNDetectedObjectObservation(boundingBox: r)
        }
    } else if !live.isEmpty {
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
    var rows: [[Double]] = []
    for (id, o) in live.sorted(by: { $0.key < $1.key }) {
        let b = o.boundingBox
        rows.append([Double(id), Double(b.midX * W), Double((1 - b.midY) * H), Double(b.width * W), Double(b.height * H),
                     Double(o.confidence)])
    }
    let line: [String: Any] = ["t": t, "tracks": rows]
    if let d = try? JSONSerialization.data(withJSONObject: line), let s = String(data: d, encoding: .utf8) {
        print(s)
    }
    t += 1
}
