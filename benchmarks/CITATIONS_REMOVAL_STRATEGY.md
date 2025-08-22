# Strategic Citation Removal for Balance

## Summary
Added 12 new journal-specific citations (+84 lines to bibliography)
Need to remove 5-6 older/redundant citations to maintain balance

## Candidates for Removal (Based on Age and Relevance)

### 1. Very Old General Papers (2015-2017)
- `gongal2015sensors` (2015) - Very old sensor paper, superseded by recent advances
- `aravind2017task` (2017) - Old task-based approach, less relevant for modern motion planning
- `zhang2016development` (2016) - Early development paper, methodology outdated
- `cubero2016automated` (2016) - Old automation approach, not motion planning focused

### 2. Redundant Survey Papers (Multiple Reviews on Similar Topics)
- `zhao2016review` (2016) - Old review of vision techniques, covered by newer surveys
- `narvaez2017survey` (2017) - Early survey of harvesting robots, superseded by recent work

### 3. Less Relevant Papers for Motion Planning Focus
- `vasconez2019human` (2019) - Human-robot interaction focus, not motion planning core
- `mavridou2019machine` (2019) - General machine learning, not specifically motion planning

## Recommended Removals (6 citations total)

1. **gongal2015sensors** - 2015 sensor paper (very old)
2. **zhao2016review** - 2016 vision review (redundant with newer surveys)  
3. **aravind2017task** - 2017 task-based approach (outdated methodology)
4. **zhang2016development** - 2016 development paper (superseded)
5. **cubero2016automated** - 2016 automation (not motion planning focused)
6. **vasconez2019human** - 2019 human-robot interaction (tangential to motion planning)

## Impact Analysis
- **Removes**: 6 older citations (2015-2019)
- **Net Change**: +6 citations (12 added - 6 removed)
- **Quality Improvement**: Replaced older general papers with recent journal-specific papers
- **Journal Respect**: 3 citations from each target journal showing current awareness

## Implementation Strategy
1. Remove these 6 citations from all ref.bib files
2. Check if any removed citations are actually used in text
3. Replace any critical content with references to new citations
4. Test compilation across all versions
5. Document final citation changes

## Balance Achievement
- **Before**: ~380 citations (many older general papers)
- **After**: ~386 citations (with recent journal-specific papers)
- **Quality**: Higher relevance and journal alignment
- **Recency**: Better representation of 2024-2025 state-of-the-art