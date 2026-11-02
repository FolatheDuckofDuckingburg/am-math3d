#!/bin/sh
# Build the native parts (macOS, Apple silicon): Apple Vision's tracker, subject mask and instances, the given-box tracker, and
# AM's eyes in C.
set -e
cd "$(dirname "$0")"
swiftc -O am/video_eyes.swift -o am/video_eyes
clang -O2 -fwrapv -shared -o am/libam_eyes.dylib am/am_eyes.c
swiftc -O math3d/track_from.swift -o math3d/track_from
swiftc -O math3d/subject_mask.swift -o math3d/subject_mask
swiftc -O math3d/instances.swift -o math3d/instances
echo "built am/video_eyes, am/libam_eyes.dylib, math3d/track_from, math3d/subject_mask, math3d/instances"
