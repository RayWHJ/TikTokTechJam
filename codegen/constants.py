"""Frozen constants shared across codegen/ (single source of truth)."""

# Every column in KuaiRand-Pure/data/video_features_statistic_pure.csv except
# video_id. These are aggregate OUTCOME statistics — not point-in-time-safe —
# and must never be model inputs unless explicitly marked point_in_time=True.
NON_CAUSAL_COLUMNS = [
    "show_cnt", "show_user_num", "play_cnt", "play_user_num", "play_duration",
    "complete_play_cnt", "complete_play_user_num", "valid_play_cnt",
    "valid_play_user_num", "long_time_play_cnt", "long_time_play_user_num",
    "short_time_play_cnt", "short_time_play_user_num", "play_progress",
    "comment_stay_duration", "like_cnt", "like_user_num", "click_like_cnt",
    "double_click_cnt", "cancel_like_cnt", "cancel_like_user_num", "comment_cnt",
    "comment_user_num", "direct_comment_cnt", "reply_comment_cnt",
    "delete_comment_cnt", "delete_comment_user_num", "comment_like_cnt",
    "comment_like_user_num", "follow_cnt", "follow_user_num", "cancel_follow_cnt",
    "cancel_follow_user_num", "share_cnt", "share_user_num", "download_cnt",
    "download_user_num", "report_cnt", "report_user_num", "reduce_similar_cnt",
    "reduce_similar_user_num", "collect_cnt", "collect_user_num",
    "cancel_collect_cnt", "cancel_collect_user_num", "direct_comment_user_num",
    "reply_comment_user_num", "share_all_cnt", "share_all_user_num",
    "outsite_share_all_cnt",
]

# Same-row auxiliary interaction signals. Allowed ONLY as auxiliary loss targets,
# never as input feature arrays fed into the model.
AUXILIARY_SIGNALS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "play_time_ms",
]

# Oracle ceiling for the primary metric on this dataset (from the starter kit
# README): a validation/test primary above this is physically impossible and is
# strong evidence of a leak.
ORACLE_PRIMARY_CEILING = 0.8645
FM_BASELINE_TEST_PRIMARY = 0.5946

# target_component values that mean "edit data.py's feature encoding".
FEATURE_COMPONENTS = {
    "feature", "features", "feature_encoding", "encoding", "data", "data.py",
    "history", "sequence", "user_history", "auxiliary", "aux", "auxiliary_signal",
    "field", "fields", "embedding_input", "context_feature",
}
