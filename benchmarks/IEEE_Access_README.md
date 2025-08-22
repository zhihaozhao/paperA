# Journal Template Requirements Analysis

## Root Cause of Compilation Errors

The LaTeX compilation errors occurred because I attempted to use a single Elsevier template (`cas-dc.cls`) for all journals, which have completely different document classes and command structures.

## Required Official Templates by Journal

### 1. IEEE Access
- **Required**: `IEEEtran.cls` from IEEE
- **Download**: https://www.ieee.org/publications/authors/author-templates.html
- **Document Class**: `\documentclass[journal]{IEEEtran}`
- **Author Format**: `\IEEEauthorblockN{}` and `\IEEEauthorblockA{}`
- **Keywords**: `\begin{IEEEkeywords}...\end{IEEEkeywords}`
- **Bibliography**: `\bibliographystyle{IEEEtran}`

### 2. SN Applied Sciences (Springer)
- **Required**: `sn-jnl.cls` from Springer Nature
- **Download**: https://resource-cms.springernature.com/springer-cms/rest/v1/content/19242230/data/v1
- **Document Class**: `\documentclass[sn-basic]{sn-jnl}`
- **Author Format**: `\author{Name\Email{email}\Orcid{orcid}}`
- **Abstract**: `\abstract{Background... Methods... Results... Conclusions...}`
- **Keywords**: `\keywords{keyword1, keyword2, ...}`

### 3. Robotics and Autonomous Systems (Elsevier)
- **Required**: `cas-dc.cls` from Elsevier ✅ CORRECT
- **Document Class**: `\documentclass[a4paper,fleqn]{cas-dc}`
- **Author Format**: `\author[1]{Name}` with `\address[1]{Affiliation}`
- **Highlights**: `\begin{highlights}...\end{highlights}`
- **Bibliography**: `\bibliographystyle{elsarticle-num}`

### 4. Discover Robotics (Springer Nature)
- **Required**: `sn-jnl.cls` from Springer Nature (same as SN Applied Sciences)
- **Document Class**: `\documentclass[sn-basic]{sn-jnl}`
- **Author Format**: Springer format with `\Email{}` and `\Orcid{}`
- **Focus**: Innovation and discovery emphasis

## Current Status
- **RAS**: Correct template (Elsevier), just needs cleanup
- **Others**: Need proper journal-specific `.cls` files to compile correctly

## Recommended Action
1. Download official templates from each journal's website
2. Migrate content to proper template structure
3. Use journal-specific commands and formatting
4. Test compilation with proper document classes

## Alternative: Generic Fallback
If official templates unavailable, use standard document classes:
- IEEE: `\documentclass{article}` with IEEE-style formatting
- Springer: `\documentclass{article}` with Springer-style formatting
- Keep content identical, just adapt structure