# row anchors are a series of pre-defined coordinates in image height to detect lanes
# the row anchors are defined according to the evaluation protocol of CULane and Tusimple
# since our method will resize the image to 288x800 for training, the row anchors are defined with the height of 288
# you can modify these row anchors according to your training image resolution

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]

# culane_row_anchor = [50, 71, 92, 113, 134, 155, 176, 197, 218, 239, 260, 281, 302, 323, 344, 365, 386, 407, 428, 449]
culane_row_anchor = [50, 61, 72, 83, 94, 105, 116, 127, 138, 149, 160, 171, 182, 193, 204, 215, 226, 237, 248, 259, 270, 281, 292, 303, 314, 325, 336, 347, 358, 369, 380, 391, 402, 413, 424, 435, 446]
