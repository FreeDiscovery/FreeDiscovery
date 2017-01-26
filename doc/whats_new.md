# Release history

## Version 0.8

**Feb. 25, 2017**

### New features  

 * NearestNeighbor search now can perform categorization both with positive/negative training documents (supervised) as well as with only a list of positive documents (unsupervised) (PR [#50](https://github.com/FreeDiscovery/FreeDiscovery/pull/50))
 * Document search and semantic search (PR [#63](https://github.com/FreeDiscovery/FreeDiscovery/pull/63))


### Enhancements
 
 * In depth code reorganisation making each processing step a separate REST endpoint (PR [#57](https://github.com/FreeDiscovery/FreeDiscovery/pull/57))
 * More rubust default parameters (PR [#65](https://github.com/FreeDiscovery/FreeDiscovery/pull/65))

### API Changes
 
 * Ability to associate external `document_id`, `rendition_id` fields when ingesting documents; document categorization can now be used with these external ids. 
 * All the wrappers classes are made private in the Python API
 * The same categorization and clustering enpoints can operate ether in the document-term space or in the LSI space (PR [#57](https://github.com/FreeDiscovery/FreeDiscovery/pull/57))
