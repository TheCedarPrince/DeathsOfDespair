"""
```julia
download_arda(output_dir; 
    layer = "state", 
    year = 2020
)
```

Downloads religious congregational membership data from ARDA (Association of Religion Data Archives) for the specified geographic layer and year, saving the data as CSV files.

## Arguments

- `output_dir::String`: The directory where downloaded CSV files will be saved.

## Keyword Arguments

- `layer::String="state"` (optional): The geographic level of the data. 

- `year::Int=2020` (optional): The year of the data to retrieve (e.g., 2020).

## Returns

- This function does not return anything

## Example

```julia
download_arda("./data", layer="state", year=2020)
```

This will download state-level congregational membership data for 2020 and store the CSV files in `./data`.
"""
function download_arda(output_dir ; layer = "state", year = 2020)

    base_url = "https://www.thearda.com/us-religion/census/congregational-membership"

    @info "Beginning download of $layer data for year $year."

    if layer == "state"
        
        for state in keys(FIPS_CODES)

            @info "Downloading data for $(FIPS_CODES[state])"

            url = base_url * "?t=1&y=$(year)&y2=0&c=$(state < 10 ? "0$(state)" : state)"

            try

                html = read_html(url);
                tables = html_elements(html, ["body", "div", "table"]);
                data = tables[2] |> html_table;

                file_name = "$(layer)_$(FIPS_CODES[state] |> lowercase).csv"

                CSV.write(joinpath(output_dir, file_name), data)

            catch

                @warn "Data not available for $(FIPS_CODES[state]); continuing."

            end
        end

    end

    @info "Requested data has been downloaded! 🎉";

end

export download_arda
