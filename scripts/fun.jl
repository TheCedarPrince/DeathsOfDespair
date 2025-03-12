using DrWatson
@quickactivate "DeathsOfDespair"

using DataFrames

df = DataFrame(:ID => 1:10, :Data => rand(10))
