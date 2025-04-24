using MLJ
using CairoMakie

LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0

RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels verbosity=0

final_data = CSV.read(datadir("exp_pro", "arda_ipums_census.csv"), DataFrame);

data = leftjoin(final_data, df, on = :STATE => :LocationDesc)

cleaned = data[:, Not(
  :Topic, 
  :REGION,
  :DIVISION,
  :Tradition,
  :geometry)
] |> x -> sort!(x, :STATE) |> unique |> dropmissing

coerce!(cleaned, Count => Continuous)

rust_belt = ["Illinois", "Indiana", "Michigan", "Missouri", "New York", "Ohio", "Pennsylvania", "West Virginia"]

rust_df = filter(row -> in(row.STATE, rust_belt), cleaned)[:, Not(:STATE)]
non_rust_df = filter(row -> !in(row.STATE, rust_belt), cleaned)[:, Not(:STATE)]

mach_linear = fit!(machine(LinearRegressor(), non_rust_df[:, Not(:DataValue)], non_rust_df.DataValue));
predictions_linear = MLJ.predict(mach_linear, rust_df)
results_linear = MLJ.evaluate!(mach_linear, resampling = CV(; nfolds = 10))

mach_rr = fit!(machine(RidgeRegressor(), non_rust_df[:, Not(:DataValue)], non_rust_df.DataValue));
predictions_rr = MLJ.predict(mach_rr, rust_df)
results_rr = MLJ.evaluate!(mach_rr, resampling = CV(; nfolds = 10))

fig = Figure();
ax = Axis(fig[1, 1], title="L2-Loss per Fold", xlabel="Fold", ylabel="Loss")

scatterlines!(ax, 1:10, results_linear.per_fold[1], color=:blue, marker=:circle, markersize=10, label = "Linear Regression")
scatterlines!(ax, 1:10, results_rr.per_fold[1], color=:red, marker=:circle, markersize=10, label = "Ridge Regression")
axislegend(position = :rb, labelsize = 15)

save(plotsdir("cross_validation.png"), fig)
