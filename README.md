Roll Number - 23CS91R06
Name - Naquee Rizwan

# Information Retrieval

# Dataset location

-- Be sure that Cran folder contains all 3 files with the exact names as follows :-
- cran.all.1400
- cran.qry
- cranqrel

# Code information

**----------Assignment2_23CS91R06_ranker.py----------**

-- [Command to run] --
- python Assignment2_23CS91R06_ranker.py "path_to_CRAN_folder" "path_to_model_queries_binary_file"

-- [Example] --
- Assuming the location of CRAN folder = "cran" and model queries binary file = "model_queries_23CS91R06.bin"
- python Assignment2_23CS91R06_ranker.py "cran" "model_queries_23CS91R06.bin"

**----------Assignment2_23CS91R06_evaluator.py----------**

-- [Command to run] --
- python Assignment2_23CS91R06_evaluator.py "path_to_gold_standard_ranked_list" "path_to_ranked_output_file"

-- [Example] --
- Assuming the path of gold standard ranked list = "cran/cranqrel" and path of ranked text file as per lnc.ltc scheme = "Assignment2_23CS91R06_ranked_list_A.txt"
- python Assignment2_23CS91R06_evaluator.py "cran/cranqrel" "Assignment2_23CS91R06_ranked_list_A.txt"
- Ranked text file's name can only be one amongst these - ["Assignment2_23CS91R06_ranked_list_A.txt", "Assignment2_23CS91R06_ranked_list_B.txt", "Assignment2_23CS91R06_ranked_list_C.txt"]

# Output pattern of Assignment2_23CS91R06_metrics_A.txt, Assignment2_23CS91R06_metrics_B.txt, Assignment2_23CS91R06_metrics_C.txt in order

- Query-wise Average Precision (AP) @10
- Query-wise Average Precision (AP) @20
- Query-wise Normalized Discounted Cumulative Gain (NDCG) @10
- Query-wise Normalized Discounted Cumulative Gain (NDCG) @20
- Mean Average Precision (AP) @10
- Mean Average Precision (AP) @20
- Mean Normalized Discounted Cumulative Gain (NDCG) @10
- Mean Normalized Discounted Cumulative Gain (NDCG) @20