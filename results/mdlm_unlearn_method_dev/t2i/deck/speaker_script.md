# Speaker script: Does Concept Forgetting Survive Relearning?

Presenter: Bijay Pandey · Howard University · 2026-10-27
Read aloud as written. Approximate spoken length 13 minutes.

---

## Slide 1: Does Concept Forgetting Survive Relearning?

Thank you all for coming. I am Bijay Pandey, from Howard University, and today I want to talk about
whether a text to image model can be made to forget a concept, and whether that forgetting lasts.
The practical setting is this. A company releases an image generation model and later decides it
should no longer produce some concept. Many published methods now edit the weights so the model
stops generating it. But a released model does not stay frozen. People download it and fine tune it
on their own data. So the question that actually matters for deployment is not
only whether the concept is gone today, but whether a modest amount of later training can bring it
back. That is the question this work tries to answer, and I will show you measurements rather than
intuitions.

---

## Slide 2: Our approach: UTRAP

The method we have been developing is called UTRAP. The idea is to anticipate the attack during
training. We start from a model that has already had the concept erased. We then simulate what an
adversary would do, by running a few fine tuning steps that try to teach the concept back. We take
the model that simulated adversary produced and push the defence in the direction that undoes it,
while a preservation term keeps unrelated abilities intact. K is the number of
simulated attack steps. K equals eight is our main configuration, and K equals zero is the control,
where no attack is simulated at all. One implementation detail matters for interpreting everything
that follows. The simulated attack is run with the gradient detached, so we take a gradient at the
end point of the simulated attack and treat the path to that end point as fixed. We do not
differentiate through the attack trajectory.

---

## Slide 3: How we evaluate forgetting

We measure three things. First, target suppression before any attack, which asks whether the
concept is actually gone. Second, preservation, which asks whether unrelated concepts still work.
Third, target recovery after relearning, which asks whether the concept comes back. The recovery
number is called H. H is the fraction of fifteen generated images, from three prompts the attacker
never trained on, that a fixed CLIP classifier labels as the target concept. The denominator is
fifteen, so H moves in steps of about zero point zero six seven. Higher H after an attack means more
of the concept came back, so for a defence lower is better. Our main concept is church, with Van
Gogh style as a second concept. One point on honesty of labelling. We have official code execution
for IMMA, from paper ports for several baselines, and our own method, and I will mark which is
which on each slide rather than calling everything a reproduction.

---

## Slide 4: What forgetting looks like

These are real generated images, not illustrations. Every image uses the same prompt, the same
generator seed, the same twenty five sampling steps, the same guidance scale and the same
resolution, so the differences come from the model weights and nothing else. The top row is the
target prompt, a photo of a church. On the left the original model produces a clear church, and each
erasure method replaces it with something else. Now look at the bottom row, which is a concept
nobody tried to erase. For ESD it survives cleanly. For the other three it visibly degrades, even
though their retained concept scores sit between zero point eight seven and zero point nine seven.
That gap matters. Our preservation metric is a closed set classifier, and a textured blob is still
classified as a golf ball. The number says preserved while the image says otherwise.

---

## Slide 5: The same comparison across every method we hold

This is the same panel extended to the remaining methods in our collection, under identical
generation settings. I am showing these together because a single qualitative example can be
misleading, and because the methods differ in what they put in place of the erased concept. Some
produce an unrelated object, some produce a blurred or abstract image, and some produce something
that still has the general shape of the original scene. Two of these are worth naming. The IMMA
result here is our domain adaptation of the method onto our own teacher model, not the authors'
released setting, which I will show separately. And a few method families are deliberately absent
from this panel, because we found implementation problems in our own versions of them that mean
they would not be a fair representation of the published work.

---

## Slide 6: Initial forgetting and preservation

Now the quantitative version of what you just saw. The horizontal axis is how much of the target
concept the model still generates after erasure, so further left means more forgotten. The vertical
axis is accuracy on eight unrelated concepts, measured over one hundred and twenty images, so higher
means more preserved. I want to be careful about what that vertical axis is. It is closed set
classification accuracy on a fixed panel of concepts. It is not a measure of image quality, and the
images two slides ago showed exactly that failure: methods scoring above zero point eight seven here
were visibly degrading the concept the metric says they preserved. The
large markers are one representative build per method, and the small translucent markers are the
additional independent builds of our own method. The important observation is that the methods do
not start the later attack from the same place, so any comparison of what happens after an attack
has to account for where each method began.

---

## Slide 7: Does the concept return?

This is the central result of the talk. Each row is one method. The first column is the original
model, the second is after erasure, and the third is after a relearning attack. Within each row the
prompt, the generator seed, and all generation settings are identical across the three columns, so
the only thing changing is the weights. The attack is twenty five gradient steps of ordinary fine
tuning on images of the concept, at a learning rate I will come back to in two slides. The result is
that the concept returns. It returns for the baselines and it returns for our method. In the
measurements behind these images, recovery reaches the maximum value the metric can take for every
arm we tested once the attack is properly configured. Forgetting, as these methods implement it, did
not survive ordinary later training.

---

## Slide 8: Does attack anticipation help?

This slide isolates the part of our own method that was supposed to provide the benefit. Each marker
is one attack on one defence build, at a fixed budget of twenty five steps. Filled markers are
attacks that passed our validity check, meaning the attacker demonstrably relearned the prompts it
was trained on. Hollow markers failed that check and are inconclusive rather than successful
defences, and I am showing them rather than hiding them. On the left is our control with no attack
anticipation, in the middle is our main configuration with eight simulated attack steps, and on the
right is the untreated teacher model the two were built from. Both of our groups ran the same number
of outer updates, and the eight step group additionally received about three times as much training
data. Despite that, the configuration with attack anticipation is not distinguishable from doing
nothing to the teacher. In this setting, simulating the attacker did not help.

---

## Slide 9: Attack calibration changes the conclusion

This is the slide I would most like you to remember, because it changed our own reading of our own
results. Here the defence is held fixed and the attack budget is held fixed at twenty five steps.
The only thing that varies is the attacker's learning rate. Each marker is a single attack seed. At
the learning rate we had been using, our control arm looks resistant, sitting near the bottom of the
axis. Raise the learning rate fivefold and the same checkpoints go to full recovery. Raise it again
and recovery goes back down, so a larger learning rate is not automatically a stronger attack. Two consequences follow. A defence can appear to resist simply because the attacker was poorly
tuned, and any recovery number measured at a single learning rate is a lower bound rather than an
estimate of robustness. The honest statistic is the worst recovery observed over a sweep, and even
that is only a statement about the grid we tested.

---

## Slide 10: A published method under our evaluation: IMMA

I want to show one published method run from the authors' own code, kept separate from everything
else. This is IMMA, on Van Gogh style, at the training budget documented in the authors' repository.
The images are the original model, the ESD erased model that IMMA starts from, and the model after
IMMA. The plot shows what happened when we then attacked it. Three attacks on one defence build, all
three passing the validity check, with recovery of zero point eight seven, one point zero, and one
point zero. I want to be precise about the scope of this. We ran the authors' code, but the training
images, the evaluation prompts, and the classifier are ours. So this is an execution of published
code under our evaluation, and it is not a reproduction of the numbers in their paper, and it is not
a comparison against the church results, which were produced under different conditions.

---

## Slide 11: What we learned and the next decision

Three things. First, across every method we tested, concept forgetting did not survive a modest
relearning attack. Second, the attack anticipation mechanism in our own method produced no
measurable benefit over the untreated teacher, at matched training budgets and across three
independent builds. Third, and most important methodologically, how resistant a defence appears
depends strongly on how well the attacker is tuned, and we were ourselves misled by this before we
swept the attacker. So at present UTRAP does not demonstrate a repeatable advantage, and I am not
going to present it as one. What remains open is whether anything in this family survives an attacker tuned as carefully as the
defence. That is the next experiment. Take our strongest configuration and tune the attacker
against it with the effort we spent tuning the defence, rather than at a fixed learning rate. If the
concept still returns, this family does not deliver durable forgetting, and that is worth reporting
on its own.

---

## Slide 12: References

The references for the base model, the methods, and the evaluator are listed here, and I am happy to
take questions.
