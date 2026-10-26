"""Supported datasets. `detect_dataset` decides which profile an input belongs to."""
from .base import DatasetProfile
from .bgl import BGLProfile
from .hdfs_profile import HDFSProfile
from .openstack import OpenStackProfile

PROFILES = {p.name: p for p in (HDFSProfile(), BGLProfile(), OpenStackProfile())}


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown dataset '{name}'; supported: {', '.join(PROFILES)}") from None


def detect_dataset(lines, only=None):
    """Score the sampled lines against every profile (or just `only`).

    Returns (profile, format, confidence, scores) for the best (profile, format), where scores is
    {profile name: {format: score}}."""
    scores, best = {}, (None, None, -1.0)
    for name, profile in PROFILES.items():
        if only and name != only:
            continue
        s = profile.detect(lines)
        scores[name] = s
        fmt, conf = max(s.items(), key=lambda kv: kv[1])
        if conf > best[2]:
            best = (profile, fmt, conf)
    return best[0], best[1], best[2], scores


__all__ = ["DatasetProfile", "PROFILES", "get_profile", "detect_dataset"]
