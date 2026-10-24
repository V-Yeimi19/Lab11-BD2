from ucimlrepo import fetch_ucirepo


dry_bean = fetch_ucirepo(id=602)

X = dry_bean.data.features
y = dry_bean.data.targets


if __name__ == "__main__":
    print(dry_bean.metadata)
    print(dry_bean.variables)
