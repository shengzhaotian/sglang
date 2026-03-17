# Task Plan: SGLang NPU Model Adaptation
<!-- 
  WHAT: This is your roadmap for the entire task. Think of it as your "working memory on disk."
  WHY: After 50+ tool calls, your original goals can get forgotten. This file keeps them fresh.
  WHEN: Create this FIRST, before starting any work. Update after each phase completes.
-->

## Goal
<!-- 
  WHAT: One clear sentence describing what you're trying to achieve.
  WHY: This is your north star. Re-reading this keeps you focused on the end state.
-->
Adapt the <ModelName> model to the SGLang framework to support NPU devices

## Current Phase
<!-- 
  WHAT: Which phase you're currently working on (e.g., "Phase 1", "Phase 3").
  WHY: Quick reference for where you are in the task. Update this as you progress.
-->
Phase 1

## Phases
<!-- 
  WHAT: Break your task into 3-7 logical phases. Each phase should be completable.
  WHY: Breaking work into phases prevents overwhelm and makes progress visible.
  WHEN: Update status after completing each phase: pending → in_progress → complete
-->

### Phase 1: Preparation and Analysis
<!-- 
  WHAT: Understand what needs to be done and gather initial information.
  WHY: Starting without understanding leads to wasted effort. This phase prevents that.
-->
- [ ] Collect model information (path, type, configuration)
- [ ] Detect device environment (NPU/GPU count, model, memory)
- [ ] Call Agent 1 for model architecture analysis
- [ ] Derive parallel configuration (TP/EP/PP)
- [ ] Record analysis results in findings.md
- **Status:** in_progress
<!-- 
  STATUS VALUES:
  - pending: Not started yet
  - in_progress: Currently working on this
  - complete: Finished this phase
-->

### Phase 2: Code Adaptation
<!-- 
  WHAT: Implement the necessary code changes to support the model on NPU.
  WHY: This is where the actual adaptation happens.
-->
- [ ] Select adaptation strategy (reuse/modify/create new)
- [ ] Implement code changes (model class, configuration, initialization)
- [ ] Ensure NPU compatibility (operators, memory, parallelism)
- [ ] Record modifications in findings.md
- **Status:** pending

### Phase 3: Verification and Testing
<!-- 
  WHAT: Verify the adaptation works correctly on NPU.
  WHY: Catching issues early saves time. Document test results in progress.md.
-->
- [ ] Dummy weight verification (quick structure validation)
- [ ] Real weight verification (full functionality validation)
- [ ] Call Agent 2 to handle verification failures (if needed)
- [ ] Call Agent 3 for comprehensive testing
- [ ] Record test results in findings.md
- **Status:** pending

### Phase 4: Completion and Delivery
<!-- 
  WHAT: Final review and handoff to user.
  WHY: Ensures nothing is forgotten and deliverables are complete.
-->
- [ ] Generate model adaptation tutorial documentation
- [ ] Single signed commit (git commit -sm ...)
- [ ] Prepare handover artifacts (analysis report, operation manual, commit hash)
- [ ] Deliver final results to user
- **Status:** pending

## Key Questions
<!-- 
  WHAT: Important questions you need to answer during the task.
  WHY: These guide your research and decision-making. Answer them as you go.
-->
1. What is the model architecture type? (Dense/MoE/MoE+MLA/VLM)
2. What parallel configuration is needed? (TP/EP/PP)
3. Is there a reference implementation for this model in SGLang?
4. What aspects need attention for NPU compatibility?
5. What issues might be encountered during verification?

## Decisions Made
<!-- 
  WHAT: Technical and design decisions you've made, with the reasoning behind them.
  WHY: You'll forget why you made choices. This table helps you remember and justify decisions.
  WHEN: Update whenever you make a significant choice (technology, approach, structure).
-->
| Decision | Rationale |
|----------|-----------|
|          |           |

## Errors Encountered
<!-- 
  WHAT: Every error you encounter, what attempt number it was, and how you resolved it.
  WHY: Logging errors prevents repeating the same mistakes. This is critical for learning.
  WHEN: Add immediately when an error occurs, even if you fix it quickly.
-->
| Error | Attempt | Resolution |
|-------|---------|------------|
|       | 1       |            |

## Notes
<!-- 
  REMINDERS:
  - Update phase status as you progress: pending → in_progress → complete
  - Re-read this plan before major decisions (attention manipulation)
  - Log ALL errors - they help avoid repetition
  - Never repeat a failed action - mutate your approach instead
-->
- Update phase status: pending → in_progress → complete
- Re-read this plan before major decisions
- Record all errors to avoid repetition
- Don't repeat failed actions, adjust your approach