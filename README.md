# OpenAI to Z Challenge: Amazon Archaeological Discovery System

This project is a comprehensive, AI-powered pipeline designed to discover previously unknown archaeological sites in the Amazon basin. It leverages a multi-modal approach, combining advanced satellite data processing, AI-driven analysis, and an agentic workflow to identify and evaluate potential sites of interest.

## Overview

The system automates the entire discovery process, from generating candidate sites to producing a detailed final report. It uses a variety of public data sources, including satellite imagery, elevation models, and biodiversity records, to find subtle anomalies that may indicate human activity. An agentic research workflow, powered by large language models like GPT-4.1, iteratively analyzes each site, refines hypotheses, and calculates a confidence score for its archaeological potential.

## Key Features

-   **Multi-Source Data Acquisition**: Automatically fetches data from Sentinel-1/2, Landsat, various Digital Elevation Models (DEMs), and the Global Biodiversity Information Facility (GBIF).
-   **Advanced Anomaly Detection**: Employs statistical methods to detect subtle terrain, vegetation, and SAR anomalies.
-   **AI-Powered Site Evaluation**: Uses OpenAI models for expert-level archaeological assessment and hypothesis generation.
-   **Agentic Research Workflow**: An iterative process refines the analysis for each site, incorporating web searches and critiques to improve confidence.
-   **Comprehensive Reporting**: Generates detailed HTML reports with interactive visualizations, quantitative scores, and AI assessments for each site.
-   **Reproducibility**: Built on public data sources and open-source Python libraries, ensuring the analysis is fully reproducible.

## How to Run the System

### Prerequisites

1.  **Python**: Python 3.8 or higher is required.
2.  **Dependencies**: Install the required Python packages. A `requirements.txt` would typically be provided. Key dependencies include `rasterio`, `geopandas`, `openai`, `langgraph`, and `google-earth-engine`.
3.  **API Keys**: The system requires several API keys to function correctly. These should be set as environment variables, typically in a `.env` file.
    -   `OPENAI_API_KEY`: For accessing OpenAI models.
    -   `TAVILY_API_KEY`: For the web search agent.
    -   `OPENTOPOGRAPHY_API_KEY`: For high-resolution elevation data.
    -   `GOOGLE_EARTH_ENGINE_API_KEY`: For GEE project authentication.

### Execution

To start the archaeological expedition, run the main script from your terminal:

```bash
python "Amazon Final.py"
```

The script will execute the entire pipeline: generating candidate sites, shortlisting them, performing detailed analysis through the agentic workflow, and finally, generating the output reports and maps.

## Understanding the Output

The script creates a main run directory named `amazon_z_discovery_[TIMESTAMP]` for each execution. This directory contains all the data, results, and reports for the expedition.

```
amazon_z_discovery_YYYYMMDD_HHMMSS/
├── data/
│   ├── AMZN_SITE_ID_1/
│   │   ├── biodiversity/
│   │   ├── elevation/
│   │   ├── satellite/
│   │   ├── processed/
│   │   └── metadata/
│   ├── ... (one folder for each analyzed site)
├── logs/
│   └── amazon_discovery_[TIMESTAMP].log
├── maps/
│   ├── interactive_sites_map.html
│   └── [SITE_ID]_overview.png
├── outputs/
│   └── expedition_summary_[TIMESTAMP].txt
└── reports/
    └── amazon_discovery_report_[TIMESTAMP].html
```

### Directory Breakdown

-   **`data/`**: This directory stores all the raw and processed data for each analyzed site. Each site has its own sub-directory, named with a unique Site ID (e.g., `AMZN_1.968S_58.017W`).
    -   **`biodiversity/`**: Contains `.json` files with data on indicator species occurrences from GBIF.
    -   **`elevation/`**: Holds Digital Elevation Model (DEM) files (`.tif`) from sources like SRTM and NASA DEM.
    -   **`satellite/`**: Stores satellite imagery, organized by sensor (e.g., `landsat`, `sentinel2`).
    -   **`processed/`**: Contains derived data products like hillshade maps, slope, aspect, and vegetation indices (e.g., NDVI).
    -   **`metadata/`**: Includes a `.json` file with metadata about the data collection process for the site.

-   **`logs/`**: Contains a detailed log file (`.log`) for the run, which is useful for debugging and tracking the analysis process.

-   **`maps/`**: This directory contains visual outputs.
    -   `interactive_sites_map.html`: An interactive map created with Folium, showing the locations of all analyzed sites with popups containing key information.
    -   `[SITE_ID]_overview.png`: For each site, a composite image is generated showing different data layers like RGB imagery, hillshade, and vegetation analysis.

-   **`outputs/`**: Contains summary files from the expedition.
    -   `expedition_summary_[TIMESTAMP].txt`: A concise text file summarizing the expedition's results, including statistics and the top recommended site.

-   **`reports/`**: The final, comprehensive output of the system.
    -   `amazon_discovery_report_[TIMESTAMP].html`: A detailed HTML report that includes an executive summary, an overview of the methodology, and a detailed breakdown of the analysis for each site, including AI assessments and quantitative scores. 