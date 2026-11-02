"""What the Sora clip cost, against what AM-Video and the AM dict cost on it, from out/report.json.
  MACHINE: an owned Apple-silicon Mac at $0.03 an hour (price spread over 3 years, plus power), as in the paper.
  SORA: the last published OpenAI API list prices per second of video (sora-2 720p $0.10; sora-2-pro 720p $0.30,
    1080p $0.70), sold as clips of 4, 8 or 12 seconds.  Sources disagree on whether the API is still served.
  FRONTIER CHECK: Opus 5.5 shown every 5th frame of this clip, measured twice ($0.190 and $0.221).
    python3 cost.py out/report.json WINDOW"""
import json, sys

MAC = 0.03 / 3600
SORA = {"sora-2, 720p": 0.10, "sora-2-pro, 720p": 0.30, "sora-2-pro, 1080p": 0.70}
OPUS = (0.190472, 0.220756)
MIN_CLIP = 4


def main():
    rep = json.load(open(sys.argv[1]))
    w = rep["windows"][sys.argv[2] if len(sys.argv) > 2 else "150"]
    secs = rep["frames"] / rep["fps"]
    e = rep["eyes_stages"]
    stages = [("tracking (Apple Vision)", e["tracking"]["wall"]),
              ("depth (Depth Anything V2 Small, GPU)", e["depth"]["wall"]),
              ("AM: hives and law, 0 tokens", w["am"]["cpu"]),
              ("AM dict: rules written down", w["dict"]["cpu"]),
              ("2D projection: the dict at all %d frames" % rep["frames"], w["projection"]["cpu"])]
    print("THE CLIP: %.1f s, %d frames, %s" % (secs, rep["frames"], "x".join(map(str, rep["size"]))))
    print("\nAM-VIDEO + AM DICT, measured on this Mac")
    tot = 0.0
    for name, s in stages:
        tot += s
        print("  %-44s %8.2f s   $%.6f" % (name, s, s * MAC))
    print("  %-44s %8.2f s   $%.6f   (per minute of video $%.4f)" % ("total", tot, tot * MAC, tot * MAC * 60 / secs))
    print("  drawing the 2D video (presentation only)     %8.2f s" % w["render_wall"])
    print("\nSORA, to make this clip (list price)")
    for k, p in SORA.items():
        print("  %-20s $%.2f for %.0f s   ($%.2f per minute;  a %d-s minimum clip costs $%.2f)"
              % (k, p * secs, secs, p * 60, MIN_CLIP, p * MIN_CLIP))
    print("\nFINDING WHAT BROKE in this clip")
    print("  Opus 5.5, every 5th frame: $%.3f-$%.3f   ($%.2f-$%.2f per minute)"
          % (OPUS[0], OPUS[1], OPUS[0] * 60 / secs, OPUS[1] * 60 / secs))
    print("  AM-Video + dict:           $%.5f        ($%.4f per minute)" % (tot * MAC, tot * MAC * 60 / secs))
    st = w["stats"]
    print("\nTHE DICT: %d of %d numbers ruled (null, shuffled in time: %d of %d); boxes drawable from rules: %d of %d "
          "object-frames; rule bits %d vs %d for the measurements they replace"
          % (st["ruled"], st["numbers"], w["null"]["ruled"], w["null"]["numbers"], st["boxes_drawable_from_rules"],
             st["object_frames"], st["dict_bits"], st["raw_measurement_bits"]))
    print("  frames with a break against the dict: %d;  4-s segments with a break: %s"
          % (st["frames_with_a_break"], st["four_second_segments_with_a_break"]))


if __name__ == "__main__":
    main()
