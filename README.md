##TODO: 

Now that we have two flows - replacing the function signature vs keeping the entire generated code by the generation function, we need to make sure that replacing the function signature works with helper functions as well. We need to do the same reordering of functions and keep all functions except main and tests for replacing the function signature.


## confidence_multi_agent_generation

1. The planner agent will write pseudocode, then it will evaluate the difficulty of the problem and then decide what the termination conditions are. 

If all of the agents are extremely unconfident, the planner can restart everything with a *different* pseudocode.

2. The coder agent will code a solution with zero-shot chain of thought.
3. Then the tester agent will attempt to compile the code.
4. Then the tester agent will write tests, and depending on its confidence in the tests, it'll pick another agent to review the tests it just wrote.
5. If the tests are found to be very wrong, the tester agent is told to be less confident in its system prompt for the rest of the question.
6. The coder agent will refine the solution and repeat 3-5
