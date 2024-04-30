import sys
import pandas

df = pandas.read_fwf("train")
df2 = pandas.read_fwf("test-sample")


#print(df.sort_values("f_acid"))
print(df)
f_acid = df.median().iloc[0]

print(f_acid)


df1 = df[df["f_acid"] <= 6]
print(df1)

print(df.columns[1])