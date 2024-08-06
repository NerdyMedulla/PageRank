import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Initialize the probability distribution dictionary
    prob_dist = {}
    all_pages = corpus.keys()
    num_pages = len(all_pages)

    # If the current page has no outgoing links, distribute evenly among all pages
    if len(corpus[page]) == 0:
        for p in all_pages:
            prob_dist[p] = 1 / num_pages
        return prob_dist

    # Distribute probability for damping factor
    linked_pages = corpus[page]
    num_linked_pages = len(linked_pages)
    for p in all_pages:
        prob_dist[p] = (1 - damping_factor) / num_pages
        if p in linked_pages:
            prob_dist[p] += damping_factor / num_linked_pages

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize page rank dictionary
    page_rank = {page: 0 for page in corpus.keys()}

    # Choose the first sample at random
    current_page = random.choice(list(corpus.keys()))

    # Perform the sampling
    for _ in range(n):
        page_rank[current_page] += 1
        transition_prob = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(transition_prob.keys()), weights=transition_prob.values(), k=1)[0]

    # Normalize the page ranks
    for page in page_rank:
        page_rank[page] /= n

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    page_rank = {page: 1 / num_pages for page in corpus}
    new_page_rank = page_rank.copy()

    converged = False
    while not converged:
        converged = True
        for page in corpus:
            rank_sum = sum(page_rank[linking_page] / len(corpus[linking_page]) for linking_page in corpus if page in corpus[linking_page])
            new_rank = (1 - damping_factor) / num_pages + damping_factor * rank_sum
            if abs(new_rank - page_rank[page]) > 0.001:
                converged = False
            new_page_rank[page] = new_rank
        page_rank = new_page_rank.copy()

    return page_rank


if __name__ == "__main__":
    main()
