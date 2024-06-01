from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import uuid
from sqlalchemy import text


app = Flask(__name__)
app.config.from_object('config.Config')

db = SQLAlchemy(app)

class Token(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(36), unique=True, nullable=False)
    note = db.Column(db.Text, nullable=True)


class WeightsAverageChange(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token_id = db.Column(db.Integer, db.ForeignKey('token.id'), nullable=False)
    epoch = db.Column(db.Integer, nullable=False)
    layer = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Float, nullable=False)

class ActivationAverageChange(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token_id = db.Column(db.Integer, db.ForeignKey('token.id'), nullable=False)
    epoch = db.Column(db.Integer, nullable=False)
    layer = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Float, nullable=False)

class ActivationAverageValue(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token_id = db.Column(db.Integer, db.ForeignKey('token.id'), nullable=False)
    epoch = db.Column(db.Integer, nullable=False)
    layer = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Float, nullable=False)

class GradientAverageValue(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token_id = db.Column(db.Integer, db.ForeignKey('token.id'), nullable=False)
    epoch = db.Column(db.Integer, nullable=False)
    layer = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Float, nullable=False)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        new_token = Token(token=str(uuid.uuid4()))
        db.session.add(new_token)
        db.session.commit()
        flash('Token generated!', 'success')
        return redirect(url_for('metrics', token=new_token.token))
    tokens = Token.query.all()
    tokens_info = [{'token': token.token, 'note': token.note} for token in tokens]
    return render_template('index.html', tokens=tokens_info)

@app.route('/add_note/<token>', methods=['POST'])
def add_note(token):
    token_entry = Token.query.filter_by(token=token).first()
    if token_entry:
        data = request.json
        token_entry.note = data.get('note', '')
        db.session.commit()
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Token not found"}), 404



def get_data(token):
    token_entry = Token.query.filter_by(token=token).first()
    if token_entry:
        weights_data_raw = read_metrics_from_db(WeightsAverageChange, token_entry.id)
        activation_change_data_raw = read_metrics_from_db(ActivationAverageChange, token_entry.id)
        activation_value_data_raw = read_metrics_from_db(ActivationAverageValue, token_entry.id)
        gradient_value_data_raw = read_metrics_from_db(GradientAverageValue, token_entry.id)
        
        activation_change_data = {}
        for record in activation_change_data_raw:
            layer = record['layer']
            activation_change_data.setdefault(layer, []).append({
                'epoch': record['epoch'],
                'value': record['value']
            })

        weights_data = {}
        for record in weights_data_raw:
            layer = record['layer']
            weights_data.setdefault(layer, []).append({
                'epoch': record['epoch'],
                'value': record['value']
            })

        activation_value_data = {}
        for record in activation_value_data_raw:
            layer = record['layer']
            activation_value_data.setdefault(layer, []).append({
                'epoch': record['epoch'],
                'value': record['value']
            })

        gradient_value_data = {}
        for record in gradient_value_data_raw:
            layer = record['layer']
            gradient_value_data.setdefault(layer, []).append({
                'epoch': record['epoch'],
                'value': record['value']
            })
            
        metrics_data = {
            "WeightsAverageChange": weights_data,
            "ActivationAverageChange": activation_change_data,
            "ActivationAverageValue": activation_value_data,
            "GradientAverageValue": gradient_value_data
        }
        return metrics_data
    else:
        return None


def read_metrics_from_db(model, token_id):
    records = model.query.filter_by(token_id=token_id).all()
    return [{'epoch': r.epoch, 'layer': r.layer, 'value': r.value} if hasattr(r, 'layer') else {'epoch': r.epoch, 'value': r.value} for r in records]



@app.route('/metrics/<token>', methods=['GET', 'POST'])
def metrics(token):
    token_entry = Token.query.filter_by(token=token).first()
    if token_entry:
        output = get_data(token)
        if output is None:
            flash('Invalid token!', 'danger')
            return redirect(url_for('home'))
        return render_template('metrics.html', token=token, metrics=output, note=token_entry.note)
    else:
        return jsonify({'error': 'Invalid token'}), 404
        
@app.route('/delete_token', methods=['POST'])
def delete_token():
    data = request.json
    token = data.get('token', '')
    token_entry = Token.query.filter_by(token=token).first()
    
    if token_entry:
        db.session.delete(token_entry)
        db.session.commit()
        return jsonify({"success": True, "message": "Token deleted successfully"}), 200
    else:
        return jsonify({"success": False, "error": "Token not found"}), 404
        

@app.route('/update_data/<token>', methods=['GET'])
def update_data(token):
    output = get_data(token)
    if output is None:
        return jsonify({'error': 'Invalid token'}), 404
    return jsonify(output)

@app.route('/WeightsAverageChange/<token>', methods=['POST'])
def add_weights_average_change(token):
    token_entry = Token.query.filter_by(token=token).first()
    if token_entry:
        epoch = request.json.get('epoch')
        layer = request.json.get('layer')
        value = request.json.get('value')
        
        WeightsAverageChange.query.filter(WeightsAverageChange.token_id == token_entry.id, WeightsAverageChange.layer == layer,  WeightsAverageChange.epoch >= epoch).delete()
        db.session.commit()
        
        new_record = WeightsAverageChange(token_id=token_entry.id, epoch=epoch, layer=layer, value=value)
        db.session.add(new_record)
        db.session.commit()
        return jsonify({'message': 'Weights average change added successfully!'}), 201
    else:
        return jsonify({'error': 'Invalid token'}), 404

@app.route('/ActivationAverageChange/<token>', methods=['POST'])
def add_activation_average_change(token):
    token_entry = Token.query.filter_by(token=token).first()
    if token_entry:
        epoch = request.json.get('epoch')
        layer = request.json.get('layer')
        value = request.json.get('value')
        
        ActivationAverageChange.query.filter(ActivationAverageChange.token_id == token_entry.id, ActivationAverageChange.layer == layer, ActivationAverageChange.epoch >= epoch).delete()
        db.session.commit()
        
        new_record = ActivationAverageChange(token_id=token_entry.id, epoch=epoch, layer=layer, value=value)
        db.session.add(new_record)
        db.session.commit()
        return jsonify({'message': 'Activation average change added successfully!'}), 201
    else:
        return jsonify({'error': 'Invalid token'}), 404

@app.route('/ActivationAverageValue/<token>', methods=['POST'])
def add_activation_average_value(token):
    token_entry = Token.query.filter_by(token=token).first()
    if token_entry:
        epoch = request.json.get('epoch')
        layer = request.json.get('layer')
        value = request.json.get('value')
        
        ActivationAverageValue.query.filter(ActivationAverageValue.token_id == token_entry.id, ActivationAverageValue.layer == layer, ActivationAverageValue.epoch >= epoch).delete()
        db.session.commit()
        
        new_record = ActivationAverageValue(token_id=token_entry.id, epoch=epoch, layer=layer, value=value)
        db.session.add(new_record)
        db.session.commit()
        return jsonify({'message': 'Activation average value added successfully!'}), 201
    else:
        return jsonify({'error': 'Invalid token'}), 404

@app.route('/GradientAverageValue/<token>', methods=['POST'])
def add_gradient_average_value(token):
    token_entry = Token.query.filter_by(token=token).first()
    if token_entry:
        epoch = request.json.get('epoch')
        layer = request.json.get('layer')
        value = request.json.get('value')
        
        GradientAverageValue.query.filter(GradientAverageValue.token_id == token_entry.id, GradientAverageValue.layer == layer, GradientAverageValue.epoch >= epoch).delete()
        db.session.commit()
        
        new_record = GradientAverageValue(token_id=token_entry.id, epoch=epoch, layer=layer, value=value)
        db.session.add(new_record)
        db.session.commit()
        return jsonify({'message': 'Gradient average value added successfully!'}), 201
    else:
        return jsonify({'error': 'Invalid token'}), 404


if __name__ == '__main__':
    with app.app_context():
        # db.drop_all()
        db.create_all()
    app.run(debug=True)


