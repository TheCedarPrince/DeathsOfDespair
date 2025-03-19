---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.7
  kernelspec:
    display_name: Julia 1.10.5
    language: julia
    name: julia-1.10
---

# Advanced Visualizations

## Set-Up

```julia
using DrWatson
@quickactivate "CategoricalDataScience"
```

```julia
using CairoMakie
using CSV
using DataFrames
using Distributions
using Images
using MLJ
using MLJLinearModels
using StatsBase
using Statistics
using SwarmMakie
```

# Background 

From the description file at https://data.world/ml-research/gender-recognition-by-voice:

In order to analyze gender by voice and speech, a training database was required. A database was built using thousands of samples of male and female voices, each labeled by their gender of male or female. Voice samples were collected from the following resources:

*  [The Harvard-Haskins Database of Regularly-Timed Speech](http://nsi.wegall.net/)
*  Telecommunications & Signal Processing Laboratory (TSP) Speech Database at McGill University
*  [VoxForge Speech Corpus](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/8kHz_16bit/)
*  [Festvox CMU_ARCTIC Speech Database at Carnegie Mellon University](http://festvox.org/cmu_arctic/dbs_awb.html)

Each voice sample is stored as a .WAV file, which is then pre-processed for acoustic analysis using the specan function from the WarbleR R package. Specan measures 22 acoustic parameters on acoustic signals for which the start and end times are provided.

The output from the pre-processed WAV files were saved into a CSV file, containing 3168 rows and 21 columns (20 columns for each feature and one label column for the classification of male or female). You can download the pre-processed dataset in CSV format, using the link above
Acoustic Properties Measured

The following acoustic properties of each voice are measured:

*    __duration:__ length of signal
*    __meanfreq:__ mean frequency (in kHz)
*    __sd:__ standard deviation of frequency
*    __median:__ median frequency (in kHz)
*    __Q25:__ first quantile (in kHz)
*    __Q75:__ third quantile (in kHz)
*    __IQR:__ interquantile range (in kHz)
*    __skew:__ skewness (see note in specprop description)
*    __kurt:__ kurtosis (see note in specprop description)
*    __sp.ent:__ spectral entropy
*    __sfm:__ spectral flatness
*    __mode:__ mode frequency
*    __centroid:__ frequency centroid (see specprop)
*    __peakf:__ peak frequency (frequency with highest energy)
*    __meanfun:__ average of fundamental frequency measured across acoustic signal
*    __minfun:__ minimum fundamental frequency measured across acoustic signal
*    __maxfun:__ maximum fundamental frequency measured across acoustic signal
*    __meandom:__ average of dominant frequency measured across acoustic signal
*    __mindom:__ minimum of dominant frequency measured across acoustic signal
*    __maxdom:__ maximum of dominant frequency measured across acoustic signal
*    __dfrange:__ range of dominant frequency measured across acoustic signal
*    __modindx:__ modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range

The gender of the speaker is given in the __label__ column. 

Note, the features for duration and peak frequency (peakf) were removed from training. Duration refers to the length of the recording, which for training, is cut off at 20 seconds. Peakf was omitted from calculation due to time and CPU constraints in calculating the value. In this case, all records will have the same value for duration (20) and peak frequency (0).

## Part 1

Which two features are most indicative of gendered voice?

### Data Pre-Processing

```julia
df = CSV.read(datadir("exp_raw", "learning", "voice.csv"), DataFrame)
df = df[completecases(df), :]
df.label = df.label .|> x -> x == "male" ? 1 : 2
```

### Construct Covariance Matrix

```julia
M = cor(Matrix(df))
(n,m) = size(M)
```

### Plot Covariance Matrix

```julia
labels = replace.(names(df), "_" => " ") .|> titlecase
fig = Figure(; size = (700, 500));

ax = CairoMakie.Axis(fig[1, 1])
Makie.heatmap!(ax, M; colormap = :magma, colorrange = (-1,1))

ax.xticks = (1:m, labels)
ax.yticks = (1:m, labels)
ax.yreversed = true
ax.xticklabelrotation = π/2

Makie.Colorbar(fig[1, 2], colormap = :magma);

for i in 1:n
  for j in 1:m
    Makie.text!(ax,
                "$(round(M[i,j],digits=2))",
                position = (i,j),
                align = (:center, :center), 
                fontsize=10,
                color = :black)
  end
end

supertitle = Label(fig[0, :], "Correlation Histogram", fontsize = 30)

fig
```

### Plot Boxplot

```julia
labels = replace.(names(df), "_" => " ") .|> titlecase
fig = Figure(; size = (700, 500));

ax = CairoMakie.Axis(fig[1, 1])

x_count = size(df)[1]
for (category, colname) in enumerate(names(df))
    CairoMakie.boxplot!(ax, ones(x_count) * category, zscore(df[!, colname]), show_outliers = false)
end

ax.xticks = (1:m, labels)
ax.xticklabelrotation = π/2
ax.xlabel = "Characteristics"

supertitle = Label(fig[0, :], "Boxplot of Data", fontsize = 30)
supertitle.tellwidth = false

fig
```

### Plot Violin Plot

```julia
labels = replace.(names(df), "_" => " ") .|> titlecase
fig = Figure(; size = (700, 500));

ax = CairoMakie.Axis(fig[1, 1])

x_count = size(df)[1]
for (category, colname) in enumerate(names(df))
    CairoMakie.violin!(ax, ones(x_count) * category, zscore(df[!, colname]))
end

ax.xticks = (1:m, labels)
ax.xticklabelrotation = π/2

supertitle = Label(fig[0, :], "Violin Plots of Data", fontsize = 30)
supertitle.tellwidth = false

fig
```

### Plot Swarm Plot

```julia
labels = replace.(names(df), "_" => " ") .|> titlecase
fig = Figure(; size = (1000, 500));

ax = CairoMakie.Axis(fig[1, 1]);

x_count = size(df)[1]
for (category, colname) in enumerate(names(df))
    beeswarm!(ax, ones(x_count) * category, zscore(df[!, colname]); gutter = .4)
end

ax.xticks = (1:m, labels)
ax.xticklabelrotation = π/2

supertitle = Label(fig[0, :], "Swarm Plots of Data", fontsize = 30)
supertitle.tellwidth = false

fig
```

### Conclusion

Based on the exploratory analysis, it would appear that the variables `meanfun` and `Q25` are the two variables most highly correlated with `label`.

## Part 2

**Perform Linear Regression, Logistic Regression, and Quadratic Discriminant Analysis on the features, graphing the resulting fits. 
How does the two feature fit compare to the fit on all features?**

### Clarify Approach

In the following approach I took, the following hold:

- **Truth** - this refers to the actual true labels from the original dataset

- **Complete** - this refers to labeled data generated from the original dataset (minus the target labels)

- **Limited** - this refers to labeled data generated from the limited dataset I constructed using the two features I identified as most indicative of gendered voice 

To construct comparisons between the fit of each model and the model compared between the complete and limited datasets, I plotted a group bar plot that compares the fitted results to the true label values.
Additionally, to model performance, I recorded Log Loss for each of the models and compared them using binned plots to depict how the majority of losses skewed and to illustrate how far apart particular fits may be.

### Declare Constants

```julia
training_steps = 1000
```

### Linear Regression

```julia
X = df[:, Not(:label)]
y = df.label 
y = coerce(y, OrderedFactor)

LinearBinaryClassifier = @load LinearBinaryClassifier pkg=GLM

lbr = LinearBinaryClassifier()
complete_mach = machine(lbr, X, y)

complete_loss = []
for i in 1:training_steps
  train = sample(1:nrow(df), round(Int, nrow(df) * .8), replace=true)
  test = setdiff(1:nrow(df), train)

  output = MLJ.evaluate!(complete_mach, resampling = [(train, test)], measure=[log_loss])

  push!(complete_loss, output.measurement[1])
end
```

```julia
X = df[:, [:meanfun, :Q25]]
y = df.label
y = coerce(y, OrderedFactor)

LinearBinaryClassifier = @load LinearBinaryClassifier pkg=GLM

lbr = LinearBinaryClassifier()
limited_mach = machine(lbr, X, y)

limited_loss = []
for i in 1:training_steps
  train = sample(1:nrow(df), round(Int, nrow(df) * .8), replace=true)
  test = setdiff(1:nrow(df), train)

  output = MLJ.evaluate!(limited_mach, resampling = [(train, test)], measure=[log_loss])

  push!(limited_loss, output.measurement[1])
end
```

```julia
fig = Figure();

ax = CairoMakie.Axis(fig[1, 1]);

y = df[:, :label]

true_ones = count(==(1), y)
true_twos = count(==(2), y)

complete_ones = 
  convert(
    Vector{Int}, 
    predict_mode(
      complete_mach, 
      df[:, Not(:label)]
    )
  ) |> pred_y -> count(==(1), pred_y)
complete_twos = 
  convert(
    Vector{Int}, 
    predict_mode(
      complete_mach, 
      df[:, Not(:label)]
    )
  ) |> pred_y -> count(==(2), pred_y)

limited_ones = 
  convert(
    Vector{Int}, 
    predict_mode(
      limited_mach, 
      df[:, [:meanfun, :Q25]]
    )
  ) |> pred_y -> count(==(1), pred_y)
limited_twos = 
  convert(
    Vector{Int}, 
    predict_mode(
      limited_mach, 
      df[:, [:meanfun, :Q25]]
    )
  ) |> pred_y -> count(==(2), pred_y)


label_ones = [true_ones, complete_ones, limited_ones]
label_twos = [true_twos, complete_twos, limited_twos]

barplot!(
  [1, 1, 1, 2, 2, 2],
  vcat(label_ones, label_twos),
  dodge = [1, 2, 3, 1, 2, 3],
  color = [1, 2, 3, 1, 2, 3],
  colormap = [:dodgerblue, :tomato, :darkseagreen2]
)

ax.title = "Binary Classification Comparison\n(Linear Regression)"
ax.ylabel = "Counts"
ax.xticks = (1:2, ["Male", "Female"])

colors = [:dodgerblue, :tomato, :darkseagreen2]
labels = ["Truth", "Complete", "Limited"]
elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
title = "Dataset"
Legend(fig[1, 2], elements, labels, title)

fig
```

```julia
fig = Figure();
ax = CairoMakie.Axis(fig[1, 1]);

hist!(ax, complete_loss, label = "Complete", color = :dodgerblue, bins = 40)
hist!(ax, limited_loss, label = "Limited", color = :tomato, bins = 40)

ax.xlabel = "Log Loss Values"
ax.ylabel = "Bin Counts"

Label(fig[:, :, Top()], "Log Loss after $(training_steps) Training Steps\n(Linear Regression)", fontsize = 24)

axislegend("Dataset Used"; position = :ct)

fig
```

### Logistic Regression

```julia
X = df[:, Not(:label)]
y = df.label 
y = coerce(y, OrderedFactor)

@load LogisticClassifier pkg=MLJLinearModels

lc = LogisticClassifier()
complete_mach = machine(lc, X, y)

train = sample(1:nrow(df), round(Int, nrow(df) * .8), replace=true)
test = setdiff(1:nrow(df), train)

lams = []
complete_loss = []
for lam in 0:.00001:0.001
    lc.lambda = convert(Float64, lam)
    output = MLJ.evaluate!(complete_mach, resampling = [(train, test)], measure=[log_loss])

    push!(lams, lam)
    push!(complete_loss, output.measurement[1])
end

lc.lambda = lams[findmin(complete_loss)[2]]
complete_lambda = lc.lambda

complete_loss = []
for i in 1:training_steps
  train = sample(1:nrow(df), round(Int, nrow(df) * .8), replace=true)
  test = setdiff(1:nrow(df), train)

  output = MLJ.evaluate!(complete_mach, resampling = [(train, test)], measure=[log_loss])

  push!(complete_loss, output.measurement[1])
end
```

```julia
X = df[:, [:meanfun, :Q25]]
y = df.label
y = coerce(y, OrderedFactor)

@load LogisticClassifier pkg=MLJLinearModels

lc = LogisticClassifier()
limited_mach = machine(lc, X, y)

train = sample(1:nrow(df), round(Int, nrow(df) * .8), replace=true)
test = setdiff(1:nrow(df), train)

lams = []
limited_loss = []
for lam in 0:.000001:0.0001
    lc.lambda = convert(Float64, lam)
    output = MLJ.evaluate!(limited_mach, resampling = [(train, test)], measure=[log_loss])

    push!(lams, lam)
    push!(limited_loss, output.measurement[1])
end

lc.lambda = lams[findmin(limited_loss)[2]]
limited_lambda = lc.lambda

limited_loss = []
for i in 1:training_steps
  train = sample(1:nrow(df), round(Int, nrow(df) * .8), replace=true)
  test = setdiff(1:nrow(df), train)

  output = MLJ.evaluate!(limited_mach, resampling = [(train, test)], measure=[log_loss])

  push!(limited_loss, output.measurement[1])
end
```

```julia
fig = Figure();

ax = CairoMakie.Axis(fig[1, 1]);

y = df[:, :label]

true_ones = count(==(1), y)
true_twos = count(==(2), y)

complete_ones = 
  convert(
    Vector{Int}, 
    predict_mode(
      complete_mach, 
      df[:, Not(:label)]
    )
  ) |> pred_y -> count(==(1), pred_y)
complete_twos = 
  convert(
    Vector{Int}, 
    predict_mode(
      complete_mach, 
      df[:, Not(:label)]
    )
  ) |> pred_y -> count(==(2), pred_y)

limited_ones = 
  convert(
    Vector{Int}, 
    predict_mode(
      limited_mach, 
      df[:, [:meanfun, :Q25]]
    )
  ) |> pred_y -> count(==(1), pred_y)
limited_twos = 
  convert(
    Vector{Int}, 
    predict_mode(
      limited_mach, 
      df[:, [:meanfun, :Q25]]
    )
  ) |> pred_y -> count(==(2), pred_y)


label_ones = [true_ones, complete_ones, limited_ones]
label_twos = [true_twos, complete_twos, limited_twos]

barplot!(
  [1, 1, 1, 2, 2, 2],
  vcat(label_ones, label_twos),
  dodge = [1, 2, 3, 1, 2, 3],
  color = [1, 2, 3, 1, 2, 3],
  colormap = [:dodgerblue, :tomato, :darkseagreen2]
)

ax.title = "Binary Classification Comparison\n(Logistic Regression)"
ax.ylabel = "Counts"
ax.xticks = (1:2, ["Male", "Female"])

colors = [:dodgerblue, :tomato, :darkseagreen2]
labels = ["Truth", "Complete", "Limited"]
elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
title = "Dataset"
Legend(fig[1, 2], elements, labels, title)
fig
```

```julia
fig = Figure();
ax = CairoMakie.Axis(fig[1, 1]);

hist!(ax, complete_loss, label = "Complete", color = :dodgerblue, bins = 40)
hist!(ax, limited_loss, label = "Limited", color = :tomato, bins = 40)

ax.xlabel = "Log Loss Values"
ax.ylabel = "Bin Counts"

Label(fig[:, :, Top()], "Log Loss after $(training_steps) Training Steps\n (Logistic Regression)", fontsize = 24)

axislegend("Dataset Used"; position = :ct)

fig
```

### Quadratic Discriminant Analysis

```julia
X = df[:, Not(:label)]
y = df.label 
y = coerce(y, OrderedFactor)

BayesianQDA = @load BayesianQDA pkg=MLJScikitLearnInterface

bq = BayesianQDA()
complete_mach = machine(bq, X, y)

complete_loss = []
for i in 1:training_steps
  train = sample(1:nrow(df), round(Int, nrow(df) * .8), replace=true)
  test = setdiff(1:nrow(df), train)

  output = MLJ.evaluate!(complete_mach, resampling = [(train, test)], measure = [log_loss])

  push!(complete_loss, output.measurement[1])
end
```

```julia
X = df[:, [:meanfun, :Q25]]
y = df.label
y = coerce(y, OrderedFactor)

BayesianQDA = @load BayesianQDA pkg=MLJScikitLearnInterface

bq = BayesianQDA()
limited_mach = machine(bq, X, y)

limited_loss = []
for i in 1:training_steps
  train = sample(1:nrow(df), round(Int, nrow(df) * .8), replace=true)
  test = setdiff(1:nrow(df), train)

  output = MLJ.evaluate!(limited_mach, resampling = [(train, test)], measure = [log_loss])

  push!(limited_loss, output.measurement[1])
end
```

```julia
fig = Figure();

ax = CairoMakie.Axis(fig[1, 1]);

y = df[:, :label]

true_ones = count(==(1), y)
true_twos = count(==(2), y)

complete_ones = 
  convert(
    Vector{Int}, 
    predict_mode(
      complete_mach, 
      df[:, Not(:label)]
    )
  ) |> pred_y -> count(==(1), pred_y)
complete_twos = 
  convert(
    Vector{Int}, 
    predict_mode(
      complete_mach, 
      df[:, Not(:label)]
    )
  ) |> pred_y -> count(==(2), pred_y)

limited_ones = 
  convert(
    Vector{Int}, 
    predict_mode(
      limited_mach, 
      df[:, [:meanfun, :Q25]]
    )
  ) |> pred_y -> count(==(1), pred_y)
limited_twos = 
  convert(
    Vector{Int}, 
    predict_mode(
      limited_mach, 
      df[:, [:meanfun, :Q25]]
    )
  ) |> pred_y -> count(==(2), pred_y)


label_ones = [true_ones, complete_ones, limited_ones]
label_twos = [true_twos, complete_twos, limited_twos]

barplot!(
  [1, 1, 1, 2, 2, 2],
  vcat(label_ones, label_twos),
  dodge = [1, 2, 3, 1, 2, 3],
  color = [1, 2, 3, 1, 2, 3],
  colormap = [:dodgerblue, :tomato, :darkseagreen2]
)

ax.title = "Binary Classification Comparison\n(Quadratic Determinant Analysis)"
ax.ylabel = "Counts"
ax.xticks = (1:2, ["Male", "Female"])

colors = [:dodgerblue, :tomato, :darkseagreen2]
labels = ["Truth", "Complete", "Limited"]
elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
title = "Dataset"
Legend(fig[1, 2], elements, labels, title)

fig
```

```julia
fig = Figure();
ax = CairoMakie.Axis(fig[1, 1]);

hist!(ax, complete_loss, label = "Complete", color = :dodgerblue, bins = 40)
hist!(ax, limited_loss, label = "Limited", color = :tomato, bins = 40)

ax.xlabel = "Log Loss Values"
ax.ylabel = "Bin Counts"

Label(fig[:, :, Top()], "Log Loss after $(training_steps) Training Steps\n (Quadratic Discriminant Analysis)", fontsize = 24)

axislegend("Dataset Used"; position = :ct)

fig
```

