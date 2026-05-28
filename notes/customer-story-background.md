# Customer story (background)

The full background to the customer engagement referenced in the main README. Names and identifying details have been omitted; the company, internal personnel, internal tool names, and field-station locations are not disclosed.

## Who

A **top-3 global crop-science company**, plant-breeding division. I led technical delivery on this engagement as Director of Solution Architecture for the program.

## What they were trying to do

Use computer vision to count corn/maize plants per row in field imagery, in order to assess plant vigor and yield per plant for selecting hybrids to progress to the next breeding trial. **Yield improvements of ~3% gate whether a hybrid advances to further development**, so per-plant counting accuracy translates directly into multi-million-dollar breeding-program decisions.

## How they captured imagery

A **tractor-mounted field imaging system** captured ~600 km of maize-field imagery per season at a European field station, running through plots of genetic trial material with halogen lighting for night-time runs.

## The technical challenge

They had a working object-detection algorithm for **sunflowers** — sunflower plants don't overlap and have distinctive leaf shapes. **Maize broke the algorithm** because:

- **Overlapping plants** after the 4-leaf growth stage
- **Weed confusion** — multiple weed species growing between plants
- **Lighting variation** — day vs night, halogen vs daylight
- **Stressed plants** with color shifts (water, nutrient, disease stress)
- **Varietal color variation** — different hybrids have different leaf colors
- **Partial leaves** at image edges, plants at field boundaries
- **Variable image quality** from the moving tractor mount

## What the customer had already evaluated

Before engaging us, the customer had explicitly considered and rejected:

- **Mechanical Turk crowdsourcing** — labeling task too complex for untrained crowd, turnaround too slow for the data volume
- **In-field manual counting by humans** — works but doesn't scale with the increasing number of trials

## Three time points of interest

Imagery was captured at three growth stages, each answering a different agronomic question:

| Stage | Days after sowing (approx) | What it answers |
|---|---|---|
| **2-leaf** | ~10-14 | Germination success — did seeds germinate and survive bird/crow predation? |
| **4-leaf** | ~21-28 | Plant stand established — confirmation of survivors past early growth |
| **6-8 leaf** | ~35-45 | Vigor and yield prediction — most overlap, but most informative |

Target dataset: **3,000 images, 1,000 per time point.**

## What we delivered

A 3-class human-in-the-loop annotation web application:

- **Backend**: Java microservice on Red Hat OpenShift with SNS/SQS asynchronous messaging
- **Frontend**: Angular single-page app with zoom/pan, plant-marker placement, weed marking, and difficulty-rating feedback
- **Labeling protocol**: each image annotated by **5 distinct users** for consensus ground truth
- **Classes**: `plants`, `weeds`, `other` (debris, soil artifacts, edge-cut plants)
- **Output**: labeled dataset feeding a downstream custom detection algorithm built by the customer's data-science team

## What it cost

In time: roughly **12 months from initial scoping to a working labeling pipeline** producing usable ground truth, across multiple delivery teams.

In ongoing maintenance: a custom web app that needed hosting, patching, and operational support for as long as new imagery was being annotated.

## What this looks like on Roboflow today

See the main [README](../README.md) for the modern equivalent. The headline differences:

- Dataset versioning, annotation tooling, training, and deployment are all platform features rather than bespoke engineering
- The data flywheel (user-submitted corrections improving the model) collapses the centralized 5-user-consensus labeling project into a feedback button in the field-scout app
- Architecture choice (RF-DETR vs YOLOv5 etc) determines edge-deployment paths — a decision made in days rather than the months it took to scope and build a custom annotation pipeline

The point of this demo isn't that "Roboflow is faster than the bespoke version we built" — it's that **the customer conversation has fundamentally changed**. The SA isn't selling a model; they're selling a workflow that wraps the model. That's a different sales motion and a different evaluation framework.
