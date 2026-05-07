-- delete views and functions

drop function if exists delexperiment;

drop function if exists delexperiment_all;

drop function if exists delclustering;

drop function if exists delclustering_all;

drop function if exists labels_dataset_all;

drop function if exists labels_split;

drop function if exists knownlabels;

drop function if exists evaltable(q_clusteringid integer);

drop function if exists evaltable(q_clusteringid integer, q_ds_split_id integer, q_testsplits text[]);

drop function get_filtered_lemmas;

drop function if exists get_lemma_groups;

drop function if exists find_frameinstance_sentence_match;

drop function if exists show_scores;

--- 

drop view instanceassignments;

drop view if exists transitiveclusterinstances;

drop view if exists clusterinstances;

drop view if exists experiment_scores;

drop view if exists clusterings_scored;

drop view if exists frameinstances_split CASCADE;