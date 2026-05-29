## Bibliography Formatting
You are responsible for finding and formatting references cleanly. Here is the strict format:

@inproceedings{carletta2005ami,
  author    = {Mccowan, Iain and 
               Kraaij, Wessel and 
               Ashby, Simone},
  year      = {2005},
  pages     = {1--4},
  title     = {The AMI meeting corpus},
  booktitle = {Proceedings of the International Conference on Methods and Techniques in Behavioral Research (MB)},
  note      = {\url{https://researchgate.net/publication/228341280}}
}

Rules for references:
- Conference papers: 6 fields (author, year, pages, title, booktitle, note).
- Journal papers: 8 fields (author, journal, note, number, pages, title, volume, year).
- Books: 6 fields (author, year, publisher, title, note, edition).
- Authors: Separated by 'and', one per line. Full names only, no initials.
- Venues: Full name of the conference/journal with initials at the end in parenthesis (e.g., International Conference on Learning Representations (ICLR)).
- DOIs/Links: Always provide in the `note` field, wrapped in `\url{}`. Priority: short doi > doi > link.
- Missing data: Write MISSING in the corresponding entry if it cannot be found.