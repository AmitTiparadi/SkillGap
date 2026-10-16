# Evaluation and manual review

Reviewed on 2026-10-11 using the synthetic student resume in `examples/resume.txt`.
This is an exploratory check, not an accuracy benchmark or hiring assessment.

## Inputs

The fixtures are condensed paraphrases of selected requirements from the
[Enveritas backend role](https://job-boards.greenhouse.io/enveritas/jobs/4006514008)
and [SteerBridge analyst role](https://jobs.lever.co/steerbridge/31c8e973-ad10-4b76-8289-410cd1731089).
The original roles include qualifications beyond a final-year student's resume.
We did not evaluate the full postings. No private user resume was used.

## What was tested

- Text and PDF extraction, including malformed and blank PDFs.
- Controlled embedding pairs and real pinned MiniLM similarity scores.
- Mocked extraction, schema errors, one JSON retry, API transport, and cloud alias blocking.
- Threshold boundaries, exact citations, negative confirmation, and two bounded citation retries.
- Distinct-job gap frequency and required/preferred priority ties.
- End-to-end mocked reports, database writes, and FastAPI upload/privacy contracts.
- Local Ollama extraction/grading/recommendations with the same resume across both roles.

## Threshold changes

The initial thresholds were high=0.70 and low=0.40. On the included real MiniLM
pairs, the Flask API line scored 0.543 against an API requirement, the pandas
analysis line scored 0.518, and a SQL-query requirement scored 0.340. The first
two were clear support for the simple test requirements, while the SQL score
would incorrectly trigger missing at 0.40. Lowering low to 0.30 retains that
evidence as partial; lowering high to 0.50 enables LLM confirmation for the
clearer pairs. A paraphrased Flask familiarity requirement scored 0.495 after
extraction, so the final high threshold is 0.49.

These settings were tuned on a tiny development set. They are not calibrated
probabilities and should not be presented as an accuracy score. The confirmation
and citation rules remain mandatory at every high threshold.

## Baseline manual decision check

The successful 0.50/0.30 run is preserved in `*-before-adjustment.json`.
Backend: 0 covered, 6 partial, 3 missing. Analyst: 0 covered, 9 partial, 2 missing.
Both completed without structured-output fallbacks. Final rerun results are in
`backend-report.json` and `analyst-report.json`.

Every missing decision in that baseline was checked against all seven resume
lines. Docker and AWS have no evidence. The backend communication requirement
has no line demonstrating communication to a nontechnical audience. The analyst
validation and eligibility requirements also have no supporting line. These five
missing decisions agree with the resume. There were no baseline covered claims
to check; the final reports' covered claims are reviewed separately below.

| Partial decision | Manual judgment |
| --- | --- |
| Flask familiarity | Too conservative: the resume explicitly describes a Flask REST API. Lowering high to 0.49 sends this case to confirmation. |
| Extensive Python work | Reasonably cautious: a skills list and student project do not establish extensive professional work. |
| Advanced database/SQL work | Related SQL evidence exists, but advanced optimization is not demonstrated. |
| Linux/Bash | Weak partial: a Python/Git skills line does not demonstrate shell experience. |
| Professional tenure | Weak partial: being a student does not show the required years. |
| Completed degree | Partial is cautious; final-year study is not evidence of graduation. |
| Analyst trends/reporting | Related CSV analysis and charts exist, but scale, recurring reports, and dashboards are not shown. |
| ETL, predictive analytics, scheduling | Weak partial: general tools were retrieved, without evidence of these specific activities. |

The current deterministic policy assigns partial to mid-range similarities,
even when the retrieved line is only loosely related. This is a known failure
mode, retained to match the requested grading policy. A useful future experiment
is a separate related-evidence validator for mid-range scores, evaluated on
negations, tool substitutions, and tenure requirements. Do not simply lower the
low threshold further: that would increase these weak partials.

## Recommendation review

Recommendations include tasks, reasons, estimates, resource types, and conditional
resume wording. The backend and analyst task sets differ. Some estimates are
optimistic, and the model can suggest a general project for a degree requirement;
that project cannot establish the qualification. The implementation appends an
explicit qualification caveat for tenure, degree, citizenship, and clearance
gaps. A credential-aware planning node would be a useful next improvement.

## Operational failures and fixes

1. The existing 3B model failed GPU and CPU allocations on this machine. CPU
   inference and a smaller 1.5B instruction model became the defaults.
2. The 0.5B trial generated invalid quotes and recommendation JSON. Safe defaults
   marked those reports incomplete instead of claiming successful analysis.
3. Keeping PyTorch and Ollama resident exceeded the available Windows paging
   budget. Embeddings now run in a batched local worker that exits before
   grading, and Ollama unloads after each call by default. Both posting runs then
   completed without fallback warnings.
4. The environment's model-download proxy failed. The optional bootstrap prepares
   a pinned snapshot without sending document text, and offline inference uses it.
5. A NumPy/SciPy compatibility warning was resolved by pinning the tested core
   dependency versions. Some Windows test temp-directory ACLs required running
   checks with normal local permissions.
6. The live UI exposed a reused REST API key for a pandas requirement. A lexical
   grouping guard now rejects ungrounded labels and distinguishes explicit tool
   combinations and tenure. A regression test covers the unrelated-key case.
7. The small grader rejected clear Flask/pandas work and invented an SQLite
   condition. The revised prompt explicitly allows known Python library
   relationships, handles OR requirements, and prohibits adding conditions from
   the evidence. Its brief explanation is generated before the verdict.
8. Running the UI alongside the model revealed an Ollama compute-buffer
   allocation failure. A batch size of 64 resolved the same request on this
   machine and is now the default.
9. A rerun falsely marked the degree requirement covered from "Final-year
   Computer Science Engineering student" and claimed the program was completed.
   A deterministic ongoing-study guard now downgrades this confirmation to
   partial unless the line also states completion. A regression test covers it.

## Interface demonstration

The local UI uses synthetic examples. Video recording was cancelled at the
user's request; screenshots are sufficient for inspecting the interface.

## Final covered-claim check

- Backend: 1 covered, 5 partial, 3 missing. All covered citations exactly match the top retrieved resume line.
  - Reviewed requirement: Familiarity with Django or Flask. Evidence: Built a Flask REST API for a student library with SQLite and unit tests.
- Analyst: 1 covered, 8 partial, 2 missing. All covered citations exactly match the top retrieved resume line.
  - Reviewed requirement: Preferred: Python or R proficiency for data analysis.. Evidence: Skills: Python, SQL, Git, HTML, CSS

The Flask example supports familiarity with Flask; the explicit Python skills
line supports the preferred Python-or-R skill. Neither proves senior proficiency.
No missing verdict claims the person lacks a skill: it describes absent resume
evidence. Degree and tenure remain partial; short projects cannot satisfy those
qualifications. These two cases are a prototype check, not an accuracy benchmark.

## Custom interface verification

The Streamlit UI was replaced with local FastAPI and HTML/CSS/JavaScript.
All 39 Python tests pass. Browser checks covered mobile/desktop layouts, sample
loading, explicit API consent, saved JSON import, expandable evidence, and
recommendation links. The server remained responsive during CPU inference.
A live check using two short role descriptions exhausted JSON retries for both
job extractions; the interface correctly showed an incomplete report with zero
graded requirements. This highlights remaining small-model reliability limits.
Populated cards were separately checked using the existing backend evaluation
JSON through the visibly labeled saved-report import; no results were fabricated.
Suggested resume metrics must be replaced with measured outcomes.
Screenshots are in `../demo/custom-*.jpg`; no video was created.
