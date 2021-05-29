from flask import render_template, flash, redirect, url_for, request
from app import app, db, bcrypt
from app.forms import RegistrationForm, LoginForm, PostForm, SearchForm
from app.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
df = pd.read_pickle('data.pkl')
embeddings = np.array(df['embeddings'].values.tolist())


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html", title='About')


@app.route("/search", methods=['GET', 'POST'])
def search():
    form = SearchForm()
    if form.validate_on_submit():
        text = form.search.data
        vector = embed([text])
        similarity = cosine_similarity(vector, embeddings).flatten()
        idx = np.argpartition(similarity, -6)[-6:]
        results = df.iloc[idx].values
        sim = similarity[idx].copy()
        new_idx = np.argsort(-1 * sim)
        results = results[new_idx, :].tolist()
        sim = np.round(sim[new_idx] * 100, decimals=1).tolist()
        return render_template('result.html', title='Result', result=results, sim_scores=sim)
    return render_template("search.html", title='Search', form=form)


@app.route("/posts")
def posts():
    posts = Post.query.all()
    return render_template("posts.html", title='Posts', posts=posts)


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(
            form.password.data).decode('utf-8')
        user = User(username=form.username.data,
                    email=form.email.data, password=hashed_password, user_type=form.account_type.data)
        db.session.add(user)
        db.session.commit()
        flash('Accounnt created! You are nnow able to log in', 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route("/account")
@login_required
def account():
    return render_template('account.html', title='Account')


@app.route("/post/new", methods=['GET', 'POST'])
@login_required
def new_post():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(title=form.title.data,
                    content=form.content.data, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('Your post has been created!', 'success')
        return redirect(url_for('home'))
    return render_template('create_post.html', title='New Post', form=form)
