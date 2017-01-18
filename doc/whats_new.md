# Release history

## Version 0.8

**In developpement**

### New features  

 * Document search and semantic search (PR [#63](https://github.com/FreeDiscovery/FreeDiscovery/pull/63))
 * Added REST API endpoint for metrics (PR [#36](https://github.com/FreeDiscovery/FreeDiscovery/pull/36)) 

### Enhancements
 
 * In depth code reorganisation making each processing step a separate REST endpoint (PR [#57](https://github.com/FreeDiscovery/FreeDiscovery/pull/57))

### API Changes
 
 * All the wrappers classes are made private in the Python API
 * The same categorization and clustering enpoints can operate ether in the document-term space or in the LSI space (PR [#57](https://github.com/FreeDiscovery/FreeDiscovery/pull/57))
 * More rubust default parameters (PR [#65](https://github.com/FreeDiscovery/FreeDiscovery/pull/65)
