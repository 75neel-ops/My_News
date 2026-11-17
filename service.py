"""UI-independent orchestration shared by the terminal and Streamlit app."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import config
import sources as src
from models import TopicSection


def generate_sections(prefs, client, on_progress=None):
    """Fetch topic/source-group combinations concurrently and retain failures."""
    groups = [g for g in src.GROUP_ORDER if any(s.group == g for s in prefs.sources)]
    sections = {t: TopicSection(t, {g: [] for g in groups}) for t in prefs.topics}
    jobs = [(t, g, [s for s in prefs.sources if s.group == g])
            for t in prefs.topics for g in groups]

    errors = []
    error_lock = Lock()

    def run_job(topic, group, group_sources):
        try:
            return client.fetch_items(topic, group_sources, prefs), None
        except Exception as exc:
            return [], f"{topic} / {group}: {type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_job, topic, group, group_sources): (topic, group)
            for topic, group, group_sources in jobs
        }
        for n, future in enumerate(as_completed(futures), 1):
            topic, group = futures[future]
            items, error = future.result()
            sections[topic].by_group[group] = items
            if error:
                with error_lock:
                    errors.append(error)
            if on_progress:
                on_progress(n, len(jobs), topic, group, len(items))

    # Keep diagnostics available to the UI without changing the return type.
    client.fetch_errors = errors
    return [sections[t] for t in prefs.topics]
