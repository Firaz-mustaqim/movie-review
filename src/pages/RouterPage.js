import React from 'react'
import {BrowserRouter as Router, Switch, Route} from 'react-router-dom'
import Home from './home'
import Sneha from './sneha'
import App from '../about'

export default function RouterPage() {
  return (
    <div>
        <Router>
            <Switch>
                <Route path="/" exact component={Home} />
                <Route path="/s" component={Sneha} />
                <Route path="/ss" exact component={App} />
            </Switch>
        </Router>
    </div>
  )
}