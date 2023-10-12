model = LogisticRegression()

mashrooms_c = mashrooms.copy()

for col in mashrooms_c.columns: # для каждой колонки считаем оценки вероятности 
    if col != 'class':
        varoyatnost = mashrooms.groupby(col)['class'].value_counts(normalize=True).unstack().fillna(0)
        mashrooms_c[col] = mashrooms_c[['class', col]].apply(lambda x: varoyatnost[x[0]].loc[x[1]], axis=1)

        mashrooms_c[col] = np.log(mashrooms_c[col]/(1-mashrooms_c[col])) # и делаем обратное логистическое преобразования

        mashrooms_c[col] = mashrooms_c[col].apply(lambda x: 0 if x == np.inf else x) # при оценке вероятности 1 получается бесконечность, зануляем
        
X_c = mashrooms_c.drop(columns='class')
y_c = mashrooms_c['class'].copy()

y_c = LabelEncoder().fit_transform(y_c)
model.fit(X_c,y_c)
probs = model.predict_proba(X_c)[:,1]

print(probs) # с обратным логистическим преобразованием
plt.hist(probs, 50)
plt.show()
