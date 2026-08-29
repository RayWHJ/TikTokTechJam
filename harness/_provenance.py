"""Frozen non-causal column list — every column in
KuaiRand-Pure/data/video_features_statistic_pure.csv except video_id.
These are post-hoc engagement aggregates; using them as model inputs without
point_in_time=True is a label leak (they're computed over the full log, including
rows after the point being predicted).
"""

NON_CAUSAL_COLUMNS = frozenset({
    'show_cnt', 'show_user_num', 'play_cnt', 'play_user_num', 'play_duration',
    'complete_play_cnt', 'complete_play_user_num', 'valid_play_cnt', 'valid_play_user_num',
    'long_time_play_cnt', 'long_time_play_user_num', 'short_time_play_cnt',
    'short_time_play_user_num', 'play_progress', 'comment_stay_duration', 'like_cnt',
    'like_user_num', 'click_like_cnt', 'double_click_cnt', 'cancel_like_cnt',
    'cancel_like_user_num', 'comment_cnt', 'comment_user_num', 'direct_comment_cnt',
    'reply_comment_cnt', 'delete_comment_cnt', 'delete_comment_user_num',
    'comment_like_cnt', 'comment_like_user_num', 'follow_cnt', 'follow_user_num',
    'cancel_follow_cnt', 'cancel_follow_user_num', 'share_cnt', 'share_user_num',
    'download_cnt', 'download_user_num', 'report_cnt', 'report_user_num',
    'reduce_similar_cnt', 'reduce_similar_user_num', 'collect_cnt', 'collect_user_num',
    'cancel_collect_cnt', 'cancel_collect_user_num', 'direct_comment_user_num',
    'reply_comment_user_num', 'share_all_cnt', 'share_all_user_num', 'outsite_share_all_cnt',
})


def check_provenance(column_names: list, point_in_time: bool = False) -> None:
    """Raise ValueError if any name in column_names is a NON_CAUSAL_COLUMNS name,
    unless point_in_time=True.

    Args:
        column_names: candidate feature/column names to check.
        point_in_time: set True only when the caller has verified the value used
            was reconstructed as of the prediction timestamp (not the full-log
            aggregate straight from video_features_statistic_pure.csv).

    Raises:
        ValueError: listing every offending column name found.
    """
    if point_in_time:
        return
    offending = [c for c in column_names if c in NON_CAUSAL_COLUMNS]
    if offending:
        raise ValueError(
            f"non-causal column(s) used without point_in_time=True: {offending}"
        )
