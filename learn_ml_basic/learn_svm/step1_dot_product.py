# %%
import matplotlib.pyplot as plt
import numpy as np

X0, Y0, X1, Y1 = 0, 1, 2, 3

A = np.array([0, 0, 3, 2])
B = np.array([0, 0, 4, 1])

print(f"""
{A = }
{B = }
""")

# %%
data = np.stack([A, B])
vectors = data[:, [X1, Y1]] - data[:, [X0, Y0]]

a = vectors[0, :]
b = vectors[1, :]

print(f"""
{data = }
{a = }
{b = }
""")

# %%
# 基準ベクトルをbとし、単位ベクトルを作る
e = b / np.linalg.norm(b)

print(f"{e = }")

# %% [markdown]

# ## Note
# **内積と写像:** 内積$a \cdot b$についてbの方向の単位ベクトル$e=b/|b|$を用いて
# $$a \cdot b = |b|(a \cdot e)$$
# とすると、$a$の$b$方向の大きさに$|b|$をかけた量と解釈できる
# ここで、内積を$|b|$で割ることで$a$の$b$方向の成分を取り出すことができる
# $$\frac{(a \cdot b)}{|b|} = a \cdot e$$
# 図示する上では方向が必要になるため別途$e$をかける必要がある


# %%
dot_product = a @ b
proj = dot_product / np.linalg.norm(b) * e
print(f"""
{dot_product = }
{proj = }
""")

# %%

params = {
    "angles": "xy",
    "scale_units": "xy",
    "scale": 1,
}

X, Y = data[:, X0], data[:, Y0]
U, V = vectors[:, 0], vectors[:, 1]

plt.gca().set_aspect("equal")
plt.quiver(X, Y, U, V, color=["r", "b"], **params)
plt.quiver(0, 0, proj[0], proj[1], color=["g"], **params)
plt.xlim(-1, 5)
plt.ylim(-1, 3)
plt.show()

# %%
