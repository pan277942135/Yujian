# Platform UI components

The runtime is a server-rendered FastAPI/Jinja application. Reusable visual
partials live under `app/templates/platform/components/`, while this directory
marks the Platform V1 component boundary for future frontend extraction.

The shared component contract is:

- `MetricCard` — dashboard and factory KPI values;
- `StatusTag` — `SUCCESS`, `RUNNING`, `FAILED` and registry status labels;
- `BatchActionBar` — selected-image bulk actions;
- `AIResultPanel` — prediction, confidence, quality and model version;
- `ImageReviewViewer` — lazy-loaded image and accepted BBox view.

Keep these partials presentational. Mutations must call the existing service
or a `/api/platform/*` adapter; component code must not create a second data
model.
