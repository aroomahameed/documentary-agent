# Learnings

stage 1
At first stage i learn the basic hierarchy of the graph each line five me understanding of how flow work.one thing that is different that it does not relay only one state to create flow first it had Pydantic models for structure which is new thing for me then that goes to node and the interesting thing node was the researcher node the find sub topic behave differently because Fan-out (Send): the same node, many copies, run in parallel, once each and it has more condition and graph have flow of every node.one thing more interesting which i know before but i test it now that if i don't annotated it is throwing an error. LangGraph saw 5 researchers trying to write research in the same step and stopped with an error rather than silently keeping one and losing 4. The reducer tells it how to combine them (operator.add joins the lists).Another thing is why does the researcher take ResearchTask? Because each copy should only see its own subtopic. If it got the full DocState, it wouldn’t know which of the 5 subtopics was its job. Send gives each copy a small, private input.the best thing of learning so far is how to trace on langsmith each call each loop and trace back time and input output as well.
stage 2
Prompt rules get ignored; structure gets enforced. Let the LLM classify, and let code decide.
Never trust identifiers an LLM returns. Validate them, or use numbers.
Re-research can introduce new errors, so a retry limit needs a fallback.
The fact-checker can hallucinate too (the Taj Mahal “north bank”).
Stage 3
Checkpointer: saves the full state after every step, per thread_id.
interrupt(): pauses inside a node. On resume, the node restarts and interrupt() returns your answer.
Command(goto, update): a node updates the state and chooses the next node in one return.
get_state().next: empty means finished; not empty means paused (or unfinished).
Graph input: a dict starts a run, Command(resume=) answers a pause, None continues.
update_state(old_config, ...): creates a branch from any past checkpoint; that’s time travel.

## Stage 1: Research → Script

### What I built

### What broke and why

### What I'd do differently

### Question I still have
