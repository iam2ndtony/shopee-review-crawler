"""In-memory job store for live analyses: one active job at a time, nothing written to disk."""
import logging
import threading
import uuid
from dataclasses import dataclass, field

from vi_clean.predictor import ModelMissing

from .analyze import analyze_live
from .config import WebConfig
from .crawl import CrawlError, crawl, parse_product_url

log = logging.getLogger(__name__)
ACTIVE = {"waiting_login", "crawling", "analyzing"}


class Busy(Exception):
    """Another live analysis is still running."""

    def __init__(self, job_id: str):
        super().__init__(job_id)
        self.job_id = job_id


@dataclass
class Job:
    id: str
    state: str = "waiting_login"  # waiting_login, crawling, analyzing, done, error
    collected: int = 0
    error: dict | None = None
    result: dict | None = None
    ready: threading.Event = field(default_factory=threading.Event)


class JobStore:
    def __init__(self, cfg: WebConfig, crawl_fn=crawl, analyze_fn=analyze_live):
        self.cfg, self.crawl_fn, self.analyze_fn = cfg, crawl_fn, analyze_fn
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, url: str, max_reviews: int | None) -> Job:
        """ValueError for a bad link, Busy while another job is active; otherwise runs on a daemon thread."""
        shop_id, item_id = parse_product_url(url)
        limit = None if max_reviews is None else max(max_reviews, 1)  # None: every review Shopee returns
        with self._lock:
            running = next((j for j in self._jobs.values() if j.state in ACTIVE), None)
            if running:
                raise Busy(running.id)
            job = Job(id=uuid.uuid4().hex[:12])
            self._jobs[job.id] = job
            for old in [j.id for j in self._jobs.values() if j.state not in ACTIVE]:
                if len(self._jobs) <= self.cfg.max_jobs:
                    break
                del self._jobs[old]
        threading.Thread(target=self._run, args=(job, shop_id, item_id, limit), daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active(self) -> Job | None:
        """The running job, so a page that lost its id (reload, dropped poll) can pick it up again."""
        return next((j for j in self._jobs.values() if j.state in ACTIVE), None)

    def _run(self, job: Job, shop_id: str, item_id: str, limit: int | None) -> None:
        def fail(code: str, message: str) -> None:
            job.error = {"code": code, "message": message}
            job.state = "error"

        try:
            items = self.crawl_fn(shop_id, item_id, limit, self.cfg, job.ready,
                                  lambda s: setattr(job, "state", s), lambda c: setattr(job, "collected", c))
            job.state = "analyzing"
            job.result = self.analyze_fn(items)
            job.state = "done"
        except CrawlError as e:
            fail(e.code, e.message)
        except ModelMissing as e:
            fail("model_missing", str(e))
        except Exception as e:  # log the type only: messages and tracebacks could carry review text
            log.error("job %s failed: %s", job.id, type(e).__name__)
            fail("internal", "Something went wrong while analysing the reviews.")
        finally:
            # A BaseException (Playwright's greenlets raise GreenletExit) skips the handlers above; without this the
            # job would stay active forever and every later start would be refused.
            if job.state in ACTIVE:
                fail("internal", "The analysis stopped unexpectedly. Start it again.")
