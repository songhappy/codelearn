reference: https://towardsdatascience.com/7-a-b-testing-questions-and-answers-in-data-science-interviews-eee6428a8b63
1. data analysis, when to have a A/B test
2. design a A/B test
3.  sample size, 
    how long to run it,  two weeks
    how do we design the test to prevent the spillover between control and treatment? cluster people, 
        try to isolate test people from control group. time-based randomization. Basically, we select a random time, for example, a day of a week, and assign all users to either the control or treatment group.
    how to address primary effect / novelty effect? If we already have a test running and we want to analyze if there’s a novelty or primacy effect, we could 1) compare new users’ results in the control group to those in the treatment group to evaluate novelty effect 2) compare first-time users’ results with existing users’ results in the treatment group to get an actual estimate of the impact of the novelty or primacy effect.
    
3.
