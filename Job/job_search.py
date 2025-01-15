from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
import time

# Initialize the LLM with the gpt-3.5-turbo model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Function to handle rate limit with retries
async def run_with_retry(agent, retries=100, wait_time=0.5):
    """
    Run the agent with retry handling for rate limits.

    Args:
        agent: The Agent instance to run.
        retries: The maximum number of retries for rate limit errors.
        wait_time: The initial wait time between retries (in seconds).

    Returns:
        The result of the agent run if successful.

    Raises:
        Exception: If maximum retries are exceeded.
    """
    for attempt in range(retries):
        try:
            result = await agent.run()
            return result
        except Exception as e:
            # Check for rate limit error and apply incremental backoff
            if "Rate limit" in str(e):
                print(f"Rate limit hit, retrying in {wait_time} seconds (Attempt {attempt + 1}/{retries})...")
                time.sleep(wait_time)
                # Gradually increase wait time for subsequent retries
                wait_time = min(wait_time * 0.5, 10)  # Cap wait time at 60 seconds
            else:
                # Raise the error if it's not related to rate limiting
                raise e
    raise Exception("Exceeded maximum retries due to rate limits.")

# Main async function
async def main():
    task = """
    Task: Search for Web3 Product Manager job postings and save one example job to 'job_listings.txt'.
    """
    
    agent = Agent(
        task=task,
        llm=llm,
    )

    # Run the agent with retry handling for rate limits
    result = await run_with_retry(agent, retries=100, wait_time=0.5)
    
    print("Task completed. Check 'job_listings.txt' for job details.")
    print(result)

# Run the async main function
asyncio.run(main())
