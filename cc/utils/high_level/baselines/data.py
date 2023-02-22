two_segments = dict(
    train_gp=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    train_cos=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14],
    val_gp=[15, 16, 17, 18],
    val_cos=[2.5, 5.0, 7.5, 10.0],
)

data = {
    "rover": {
        "train_gp": list(range(12)),
        "train_cos": list(range(12)),
        "val_gp": list(range(12, 15)),
        "val_cos": [2.5, 4.5, 6.5],
    },
    "two_segments": two_segments,
    "two_segments_v2": two_segments,
}
