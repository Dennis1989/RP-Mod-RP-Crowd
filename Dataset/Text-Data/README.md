# PLEASE PLACE THE RP-MOD-CROWD DATASETS IN THIS FOLDER!

This is the place where the datasets that can be retrieved from [https://zenodo.org/record/5242915](https://zenodo.org/record/5242915) have to be placed.

# The `One Million Posts Corpus` Dataset

For cross-validation purposes we make use of the `One Million Posts Corpus` dataset, which we refer to as `DerStandard` dataset in our paper ().
The dataset is available via [GitHub](https://ofai.github.io/million-post-corpus/) and is further described in the following paper:

> *Dietmar Schabus, Marcin Skowron, Martin Trapp*<br>
> **One Million Posts: A Data Set of German Online Discussions**<br>
> *Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)*<br>
> *pp. 1241-1244*<br>
> *Tokyo, Japan, August 2017*

The dataset has been made available under the [`CC BY-NC-SA 4.0`](LICENSE) license.
All derivatives included in this repository are also released under the [`CC BY-NC-SA 4.0`](LICENSE) license.


## Used Data Files

The generated `CSV`-file has the following structure:

| Text                                                                                                                                                    | Value |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| Weil es dein meisten Leuten verständlicherweise vollkommen egal ist, was die Gesellschaft oder jede andere dahergelaufene Diskursgruppe von ihnen hält. | 1     |
| ...                                                                                                                                                     | ...   |
| Der Vorfall hätte nicht stattgefunden, Wenn niemand ein Kind so vernetzt hätte, sodass es los zieht, um zu morden.                                      | 0     |

The column `Text` contains the raw comments, while the column `Value` indicates whether the comment is problematic (`1`) or not (`0`).

## Selecting Data

To obtain the data you need to download the dataset in the `SQlite` format as provided [here](https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/million_post_corpus.tar.bz2).
In the following you need to import the `corpus.sqlite3` into a suitable `SQlite`-editor.

Afterwards you can run the following queries:

1.  Create a table of all problematic comments (those considered *discriminating* or *inappropriate*).
    ```sql
    CREATE TABLE Problematic AS
    SELECT * FROM Annotations_consolidated WHERE Annotations_consolidated.Category='Discriminating' AND Value > 0 GROUP BY ID_Post 
    UNION 
    SELECT * FROM Annotations_consolidated WHERE Annotations_consolidated.Category='Inappropriate' AND Value > 0 GROUP BY ID_Post;
    ```
2.  Create a table of all unproblematic comments.
    ```sql
    CREATE TABLE NonAbusive AS
    SELECT ID_Post FROM Annotations_consolidated WHERE Category = 'Discriminating' AND Value = 0
    INTERSECT
    SELECT ID_Post FROM Annotations_consolidated WHERE Category = 'Inappropriate' AND Value = 0;
    ```
3.  Create a table of all comments that are neither problematic nor unproblematic (*off-topic*).
    ```sql
    CREATE TABLE OffTopicIDs AS
    SELECT ID_POST FROM Annotations_consolidated WHERE Category='OffTopic' AND Value = 1 GROUP BY ID_Post;
    ```
4.  Filter comments marked as *off-topic* from the unproblematic ones.
    ```sql
    DELETE FROM NonAbusive WHERE NonAbusive.ID_Post IN (SELECT ID_POST FROM OffTopicIDs);
    ```
5.  Extract problematic comments as `CSV`.
    ```sql
    SELECT printf("%s %s", Headline,Body) AS Text, 1 AS Value FROM Problematic INNER JOIN Posts WHERE Posts.ID_Post = Problematic.ID_Post;
    ```
5.  Extract unproblematic comments as `CSV` (subsample to 585 to have equal quantities of problematic and unproblematic comments).
    ```sql
    SELECT * FROM (SELECT printf("%s %s", Headline,Body) AS Text, 0 AS Value FROM NonAbusive INNER JOIN Posts WHERE Posts.ID_Post = NonAbusive.ID_Post ORDER BY random() limit 585);
    ```