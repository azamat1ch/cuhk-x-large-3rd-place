# Task

You investigate candidate related non-RGB recordings and answer their manner-of-execution
questions. Candidate membership is a proposal, not established truth. "Emotion" here
means how the actions are performed. Return one offered letter per question; retain
each recording's own options and identity.

## Candidate metadata: context, not a decision

`candidate_basis` explains why these recordings were presented together:
- `numbered_overlap`: consecutive original indices and at least three shared emotion
  options supported the proposal. This is stronger metadata evidence, not visual proof.
- `question_option_overlap`: matching question structure plus shared emotion and
  action options supported the proposal without relying on numbering.
- `uncertain_numbered_question_match`: nearby original indices and matching question
  structure supported a tentative proposal; the stronger emotion-overlap rule did not.
- `supplied_membership`: an external membership file supplied the candidate; its
  reliability is unknown.
- `singleton`: only one recording is available.
Use the same visual compatibility check for every multi-recording candidate. These
labels never establish manner, completeness or correct answers. Local IDs r1/r2/r3
are arbitrary identifiers, not original recording order, chronology or speed ranks.

## Required guidance and video coverage

Use the viewing tools freely to resolve decisive differences. Read emotion-comparison
before assigning words; its resources cover matched pace, group options and subtle
manner distinctions. Read visual-inspection for source/transform guidance and
relevant action-cues when recognizing a comparable action is uncertain.

Inspect a complete chronological overview including endpoints for EVERY recording.
If IR is shorter than another permitted stream, also overview a longest source.
Finish overview pages. Use IR, plain Depth and Thermal only. Keep source clocks
separate: equal timestamps across recordings or modalities do not align events.
Never infer style from missing/frozen footage or original numbering.

## Required grouping decision

Before assigning manner words, use the required overviews to assess grouping. Return
`group_assessment.decision` and an evidence-based `reason`: compatible if person,
setting and broad action routine support comparison; incompatible if they contradict
it; uncertain if evidence is insufficient; singleton for one recording. Same person
or shared options alone do not establish compatibility. Minor omitted actions can
be actor deviations. Routine questions/options are hypotheses, not observed actions.
For incompatible or uncertain candidates, answer each recording independently in this
conversation: no sibling elimination, distinct-word assignment or relative fast/slow
prior. Explain this in comparisons and give individual alternatives. Only compatible
candidates permit joint reasoning.

## Interpretation preferences: apply to the visible evidence

These preferences guide choices between plausible words; they do not replace the
coverage and grouping requirements above. Compare total routine duration as well as
local timing: after checking comparable work and coverage, a longer execution supports
a slower/unhurried interpretation when finer cues tie. Use
emotion-comparison/references/matched-pace.md for duration comparisons and the
specific distinction between time spent grooming and the speed of individual strokes.

For compatible candidates, compare equivalent actions: movement cadence, pauses,
engagement, transitions and handling. Pace is the primary contrast to investigate.
Even compatible pairs can be incomplete. For a pair, compare both hypotheses:
a brisk/rushed versus measured contrast, or two measured performances with the fast
sibling absent. Relative speed alone does not establish haste. Choose from visible
execution, not group size or proposal strength. Triplets often contain a clear rushed take,
but their other two styles need not be different speed buckets. Use shared options
and distinct assignments only as supporting priors after compatibility is established.

Start with the clearest contrast and inspect matched intervals to resolve the closest
alternative assignment. For the closest pair, describe matched timing and handling
before attaching adjectives; common features of ordinary task performance alone
do not distinguish assignments. More than one common episode is useful when available.
Do not copy actions between siblings. Distinguish slower individual motion from
taking more time on an open-ended action: additional grooming strokes can express
unhurried engagement without a slower stroke. Apply the matched-pace guidance before
discounting longer bouts as extra work or ranking nearly equal stroke timings.
A brief hesitation can reflect recalling the script; look for repeated evidence.
If comparable episodes or visible style are inadequate, state uncertainty and still
choose the most supported offered answer. Stop when further views would repeat the
same evidence without distinguishing alternatives. No arbitrary viewing count applies.

## Required output

Return the required scene, answers, confidence, uncertainty and timeline, plus
group_assessment, comparisons and strongest_alternative. Each timeline event has its
opaque recording_id, source-clock interval, visible feature, action candidates and
delivered references.
Use the exact namespaced rN/... references returned by tools; a sibling's evidence
cannot establish this recording's motion. Comparisons should name the matched episodes
and their timeline evidence, separate movement from pauses/repetition counts, and
explain the selected words versus the strongest competing assignment. State when a
choice depends on group constraints or elimination rather than a visible distinction.
Confidence is qualitative, not a calibrated probability. No external lookup, private
feelings or unseen action should be asserted as observed evidence.
